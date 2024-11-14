######################
# Import libraries
######################
import matplotlib.pyplot as plt
from matplotlib import cm
from rdkit.Chem.Draw import SimilarityMaps
from numpy import loadtxt
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import pickle
from PIL import Image
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import PandasTools
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import joblib
from IPython.display import HTML
from molvs import standardize_smiles
from math import pi
import zipfile
import base64
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


######################
# Page Title
######################
st.write("<h1 style='text-align: center; color: #00008B;'> HDAC3_VS_assistant</h1>", unsafe_allow_html=True)
st.write("<h3 style='text-align: center; color: black;'> Virtual screening HDAC3 inhibitors by QSAR models.</h1>", unsafe_allow_html=True)
if st.sidebar.button('Application description'):
    st.sidebar.write('The HDAC3_VS_assistant application provides an alternative method for assessing the potential of chemicals to be Histone deacetylas 3 (HDAC3) inhibitors. The application also allows to predict the level of toxicity (rat, oral, LD50) of the studied compounds.  This application makes predictions based on Quantitative Structure-Activity Relationship (QSAR) models build on curated datasets. If experimental activity or toxicity values are available for the compound, they are displayed in the summary table. The  models were developed using open-source chemical descriptors based on Morgan fingerprints, along with the Gradient Boosting, Support Vector Machines  algorithms, using Python. The models were generated applying the best practices for QSAR model development and validation widely accepted by the community. The applicability domain (AD) of the models was calculated as Dcutoff = ⟨D⟩ + Zs, where «Z» is a similarity threshold parameter defined by a user (0.5 in this study) and «⟨D⟩» and «s» are the average and standard deviation, respectively, of all Euclidian distances in the multidimensional descriptor space between each compound and its nearest neighbors for all compounds in the training set. Batch processing is available through https://github.com/ovttiras/HDAC3_VS_assistant.')


with open("manual.pdf", "rb") as file:
    btn=st.sidebar.download_button(
    label="Click to download brief manual",
    data=file,
    file_name="manual of HDAC3_VS_assistant.pdf",
    mime="application/octet-stream"
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

 # Download experimental data
df = pd.read_csv('datasets/HDAC3_exp_data_inchi.csv')
res = (df.groupby("inchi").apply(lambda x: x.drop(columns="inchi").to_dict("records")).to_dict())
df_tox = pd.read_csv('datasets/mouse_intravenous_LD50_inchi.csv')
res_tox = (df_tox.groupby("inchi").apply(lambda x: x.drop(columns="inchi").to_dict("records")).to_dict())


def rdkit_numpy_convert(f_vs):
    output = []
    for f in f_vs:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        output.append(arr)
        return np.asarray(output) 

def calcfp(mol,funcFPInfo=dict(radius=2, nBits=1024, useFeatures=False, useChirality=False)):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, **funcFPInfo)
    fp = pd.Series(np.asarray(fp))
    fp = fp.add_prefix('Bit_')
    return fp

# LOAD MODELS
# HDAC3 activity models
with zipfile.ZipFile('Models/HDAC3_GBR_MF_final_FS.zip', 'r') as zip_file_gbr:
    zf_gbr=zip_file_gbr.extract('HDAC3_GBR_MF_final_FS.pkl', '.')
load_model_GBR=joblib.load(zf_gbr)

# Toxicity models
with zipfile.ZipFile('Models/Toxicity/LD50_mouse_intravenous_SVM_MF.zip', 'r') as zip_file_svm_tox:
    zf_tox=zip_file_svm_tox.extract('LD50_mouse_intravenous_SVM_MF.pkl', '.')
load_model_SVM_tox=joblib.load(zf_tox)

with zipfile.ZipFile('Models/Toxicity/LD50_mouse_intravenous_GBR_MFP.zip', 'r') as zip_file_gbr_tox:
    zf_tox_gbr=zip_file_gbr_tox.extract('LD50_mouse_intravenous_GBR_MFP.pkl', '.')
load_model_GBR_tox=joblib.load(zf_tox_gbr)

# load numpy array from csv file for hdac activity
zf_hdac = zipfile.ZipFile('Models/data3.zip') 
df_hdac = pd.read_csv(zf_hdac.open('data3.csv'))
x_tr=df_hdac.to_numpy()
model_AD_limit = 2.78
#load reduced descriptor
from pathlib import Path
path = Path('Models/feature_name_rfecv_MF.txt')
feature_name_rfecv_MF = path.read_text().splitlines()

# load numpy array from csv file for toxicity

zf = zipfile.ZipFile('Models/Toxicity/x_tr_mouse_intravenous_.zip') 
df_tox = pd.read_csv(zf.open('x_tr_mouse_intravenous_.csv'))
x_tr_tox=df_tox.to_numpy()
model_AD_limit_tox = 4.19

files_option = st.selectbox('Select input molecular files', ('SMILES', '*CSV file containing SMILES', 'MDL multiple SD file (*.sdf)'))

if files_option == 'SMILES':
    SMILES_input = ""
    compound_smiles = st.text_area("Enter only one structure as a SMILES", SMILES_input)
    if len(compound_smiles)!=0:
        canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(compound_smiles),isomericSmiles = False)
        smiles=standardize_smiles(canon_smi)
        m = Chem.MolFromSmiles(smiles)
        inchi = str(Chem.MolToInchi(m))
        im = Draw.MolToImage(m)
        st.image(im)
    
    
    if st.button('Run predictions!'):
        # Calculate molecular descriptors        
        desc_ws = calcfp(m)
        f_vs_red=desc_ws[feature_name_rfecv_MF]
        X_red = f_vs_red.to_numpy()
        X_red=X_red.reshape(1, -1)

        f_vs=desc_ws
        X = f_vs.to_numpy()
        X=X.reshape(1, -1)
        # HDAC activity
        # search experimental value
        if inchi in res:
           exp=round(res[inchi][0]['pchembl_value_mean'],2)           
           std=round(res[inchi][0]['pchembl_value_std'],4)
           chembl_id=str(res[inchi][0]['molecule_chembl_id'])
           y_pred_con='see experimental value'
           cpd_AD_vs='-'
            
        else:
            # predict activity
            prediction_RF = load_model_GBR.predict(X_red)
            y_pred_con=prediction_RF
            y_pred_con=round((y_pred_con[0]), 3)
                                    
            # Estimination AD
            neighbors_k_vs = pairwise_distances(x_tr, Y=X_red, n_jobs=-1)
            neighbors_k_vs.sort(0)
            similarity_vs = neighbors_k_vs
            cpd_value_vs = similarity_vs[0, :]
            cpd_AD_vs = np.where(cpd_value_vs <= model_AD_limit, "Inside AD", "Outside AD")
           
            # result
            cpd_AD_vs=cpd_AD_vs[0]
            exp="-"
            std="-"
            chembl_id="not detected"
        
        # Toxicity
       
        # search experimental toxicity value
        if inchi in res_tox:
           exp_tox=str(res_tox[inchi][0]['TOX_VALUE'])
           cas_id=str(res_tox[inchi][0]['CAS_Number'])
           value_ped_tox='see experimental value'
           cpd_AD_vs_tox='-'
            
        else:
             #Predict toxicity
            prediction_SVM_tox = load_model_SVM_tox.predict(X)
            prediction_GBR_tox = load_model_GBR_tox.predict(X)
            y_pred_con_tox=(prediction_SVM_tox+prediction_GBR_tox)/2
            y_pred_con_tox_t=y_pred_con_tox[0]
            MolWt=ExactMolWt(Chem.MolFromSmiles(smiles))
            value_ped_tox=str(round((10**(y_pred_con_tox_t*-1)*1000)*MolWt, 4))
             # Estimination AD for toxicity
            neighbors_k_vs_tox = pairwise_distances(x_tr_tox, Y=X, n_jobs=-1)
            neighbors_k_vs_tox.sort(0)
            similarity_vs_tox = neighbors_k_vs_tox
            cpd_value_vs_tox = similarity_vs_tox[0, :]
            cpd_AD_vs_tox = np.where(cpd_value_vs_tox <= model_AD_limit_tox, "Inside AD", "Outside AD")
            exp_tox="-"
            cas_id="not detected"
            
        st.header('**Prediction results:**')    
        common_inf = pd.DataFrame({'SMILES':smiles, 'Predicted value, pIC50': y_pred_con, 'Applicability domain': cpd_AD_vs,'Experimental value, pIC50': exp,'Standard deviation': std,
            'Chemble ID': chembl_id,
            'Predicted value toxicity, Ld50, mg/kg': value_ped_tox,
            'Applicability domain_tox': cpd_AD_vs_tox,
            'Experimental value toxicity, Ld50': exp_tox,
            'CAS number': cas_id}, index=[1])
        predictions_pred=common_inf.astype(str) 
        st.dataframe(predictions_pred)

              
        # Search for substructures and calculation of the Tanimoto index
        substructures_df = pd.read_csv("datasets/vip_substructures.csv", sep="\s+")
        # Converting SMARTS substructures into RDKit molecules
        substructure_mols = [(row['name'], Chem.MolFromSmarts(row['smarts'])) for _, row in substructures_df.iterrows()]
        # Creating a topological fingerprint for the original molecule
        mol_fp = FingerprintMols.FingerprintMol(m)

        # A dictionary for found substructures with their atomic indexes
        found_substructures = {}

        for name, substructure in substructure_mols:
            if substructure:
                match = m.GetSubstructMatch(substructure)
                if match:
                    found_substructures[name] = match
        # Checking if substructures are found
        st.header('*The most significant sub-structures that increase the inhibitory activity of HDAC3*')
        if found_substructures:
            # A passage through each found substructure and a display of a molecule with isolated atoms
            for name, atoms in found_substructures.items():
                st.write(f"The found fragment: {name}")

                # Calculating the Tanimoto coefficient
                substructure_mol = Chem.MolFromSmarts(substructures_df[substructures_df['name'] == name]['smarts'].values[0])
                sub_fp = FingerprintMols.FingerprintMol(substructure_mol)
                tanimoto_similarity = DataStructs.TanimotoSimilarity(mol_fp, sub_fp)
                st.write(f"Tanimoto coefficient: {tanimoto_similarity:.2f}")

                # visualization of a molecule with a highlighted sub-structure
                img = Draw.MolToImage(m, highlightAtoms=atoms, size=(300, 300))
                st.image(img)  # Display an image
        else:
            st.write(f"Substructures are not found in the molecule.")

        # The Brenk filter
        substructures_df = pd.read_csv("datasets/unwanted_substructures.csv", sep="\s+")
        # Converting SMARTS substructures into RDKit molecules
        substructure_mols = [(row['name'], Chem.MolFromSmarts(row['smarts'])) for _, row in substructures_df.iterrows()]

        # A dictionary for found substructures with their atomic indexes
        found_substructures = {}

        for name, substructure in substructure_mols:
            if substructure:
                match = m.GetSubstructMatch(substructure)
                if match:
                    found_substructures[name] = match
        # Checking if substructures are found        
        st.header('*The Structural Alerts or Brenk filters [DOI:10.1002/cmdc.200700139] contain substructures with undesirable pharmacokinetics or toxicity*')
        if found_substructures:
            # A passage through each found substructure and a display of a molecule with isolated atoms
            for name, atoms in found_substructures.items():
                st.write(f"The found fragment: {name}")
                # visualization of a molecule with a highlighted sub-structure
                img = Draw.MolToImage(m, highlightAtoms=atoms, size=(300, 300))
                st.image(img)  # Display an image
        else:
            st.write(f"Substructures are not found in the molecule.")  

        # The PAINS filter
        substructures_df = pd.read_csv("datasets/PAINS.csv", sep="\s+")
        # Converting SMARTS substructures into RDKit molecules
        substructure_mols = [(row['name'], Chem.MolFromSmarts(row['smarts'])) for _, row in substructures_df.iterrows()]

        # A dictionary for found substructures with their atomic indexes
        found_substructures = {}

        for name, substructure in substructure_mols:
            if substructure:
                match = m.GetSubstructMatch(substructure)
                if match:
                    found_substructures[name] = match
        # Checking if substructures are found        
        st.header('*Filter for PAINS*')
        if found_substructures:
            # A passage through each found substructure and a display of a molecule with isolated atoms
            for name, atoms in found_substructures.items():
                st.write(f"The found PAINS: {name}")
                # visualization of a molecule with a highlighted sub-structure
                img = Draw.MolToImage(m, highlightAtoms=atoms, size=(300, 300))
                st.image(img)  # Display an image
        else:
            st.write(f"PAINS are not found in the molecule.")    

if files_option == '*CSV file containing SMILES':
     
    # Read input
    uploaded_file = st.file_uploader('The file should contain only one column with the name "SMILES"')
    if uploaded_file is not None:
        df_ws=pd.read_csv(uploaded_file, sep=';')
        count=0
        failed_mols = []
        bad_index=[]
        index=0
        for i in df_ws.SMILES: 
            index+=1           
            try:
                canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(i),isomericSmiles = False)
                df_ws.SMILES = df_ws.SMILES.replace (i, canon_smi)             
            except:
                failed_mols.append(i)
                bad_index.append(index)
                canon_smi='wrong_smiles'
                count+=1
                df_ws.SMILES = df_ws.SMILES.replace (i, canon_smi)
        st.header('CHEMICAL STRUCTURE VALIDATION AND STANDARDIZATION:')
        st.write(f'Original data: {len(df_ws)} molecules')
        st.write(f'Failed data: {count} molecules')

        if len(failed_mols)!=0:
            number =[]
            for i in range(len(failed_mols)):
                number.append(str(i+1))
            
            
            bad_molecules = pd.DataFrame({'No. failed molecule in original set': bad_index, 'SMILES of wrong structure: ': failed_mols, 'No.': number}, index=None)
            bad_molecules = bad_molecules.set_index('No.')
            st.dataframe(bad_molecules)


        moldf = []
        for i,record in enumerate(df_ws.SMILES):
            if record!='wrong_smiles':
                canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(record),isomericSmiles = False)
                standard_record = standardize_smiles(canon_smi)
                m = Chem.MolFromSmiles(standard_record)
                moldf.append(m)
        
        st.write('Kept data: ', len(moldf), 'molecules')

        if st.button('Run predictions!'):         
            # search experimental value     
            exp=[]
            std=[]
            chembl_id=[]
            y_pred_con=[]
            cpd_AD_vs=[]
            number =[]
            count=0
            struct=[]
            exp_tox=[]
            cas_id=[]
            y_pred_con_tox=[]
            cpd_AD_vs_tox=[]
            for m in moldf:
                inchi = str(Chem.MolToInchi(m))
                i=Chem.MolToSmiles(m)
                struct.append(i)
                if inchi in res:
                    exp.append(round((res[inchi][0]['pchembl_value_mean']), 2))
                    std.append(round((res[inchi][0]['pchembl_value_std']), 3))
                    chembl_id.append(str(res[inchi][0]['molecule_chembl_id']))
                    y_pred_con.append('see experimental value')
                    cpd_AD_vs.append('-')
                    count+=1         
                    number.append(count)
                                
                else:
                    # Calculate molecular descriptors
                    f_vs = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024, useFeatures=False, useChirality=False)]
                    X = rdkit_numpy_convert(f_vs)
                    #Predict activity
                    prediction_RF = load_model_GBR.predict(X)
                    prediction_SVM = load_model_SVM.predict(X)
                    prediction=(prediction_RF+prediction_SVM)/2                                  
                    y_pred_con.append(round(float(prediction),3))
                    # Estimination AD
                    neighbors_k_vs = pairwise_distances(x_tr, Y=X, n_jobs=-1)
                    neighbors_k_vs.sort(0)
                    similarity_vs = neighbors_k_vs
                    cpd_value_vs = similarity_vs[0, :]
                    cpd_AD = np.where(cpd_value_vs <= model_AD_limit, "Inside AD", "Outside AD")
                    str_a = ''.join(cpd_AD)                    
                    cpd_AD_vs.append(str_a)
                    exp.append('-')
                    std.append('-')
                    chembl_id.append('-')
                    count+=1         
                    number.append(count)

                # search experimental toxicity value
                if inchi in res_tox:
                    exp_tox.append(str(res_tox[inchi][0]['TOX_VALUE']))
                    cas_id.append(str(res_tox[inchi][0]['CAS_Number']))
                    y_pred_con_tox.append('see experimental value')
                    cpd_AD_vs_tox.append('-')
                    
                else:
                    m_t = m
                    # Calculate molecular descriptors
                    f_vs_tox = [AllChem.GetMorganFingerprintAsBitVect(m_t, radius=2, nBits=1024, useFeatures=False, useChirality=False)]
                    X_tox = rdkit_numpy_convert(f_vs_tox)
                    # Estimination AD for toxicity
                    neighbors_k_vs_tox = pairwise_distances(x_tr_tox, Y=f_vs_tox, n_jobs=-1)
                    neighbors_k_vs_tox.sort(0)
                    similarity_vs_tox = neighbors_k_vs_tox
                    cpd_value_vs_tox = similarity_vs_tox[0, :]
                    cpd_AD_vs_tox_r = np.where(cpd_value_vs_tox <= model_AD_limit_tox, "Inside AD", "Outside AD")

                    # calculate toxicity  

                    prediction_SVM_tox = load_model_SVM_tox.predict(X_tox)
                    prediction_GBR_tox = load_model_GBR_tox.predict(X_tox)
                    y_pred_tox=(prediction_SVM_tox+prediction_GBR_tox)/2                            
                    MolWt=ExactMolWt(m_t)
                    value_ped_tox=(10**(y_pred_tox*-1)*1000)*MolWt
                    value_ped_tox=round(value_ped_tox[0], 4)
                    y_pred_con_tox.append(value_ped_tox)
                    cpd_AD_vs_tox.append(cpd_AD_vs_tox_r[0])
                    exp_tox.append("-")
                    cas_id.append("not detected")  

            common_inf = pd.DataFrame({'SMILES':struct, 'Predicted value, pIC50': y_pred_con, 'Applicability domain': cpd_AD_vs,
            'Experimental value, pIC50': exp,'Standard deviation': std,
            'Chemble ID': chembl_id,'No.': number,            
            'Predicted value toxicity, Ld50, mg/kg': y_pred_con_tox,
            'Applicability domain_tox': cpd_AD_vs_tox,
            'Experimental value toxicity, Ld50': exp_tox,
            'CAS number': cas_id}, index=None)
            predictions_pred = common_inf.set_index('No.')
            predictions_pred=predictions_pred.astype(str)
                        
        
            st.dataframe(predictions_pred)           
            def convert_df(df):
                return df.to_csv().encode('utf-8')  
            csv = convert_df(predictions_pred)

            st.download_button(
                label="Download results of prediction as CSV",
                data=csv,
                file_name='Results.csv',
                mime='text/csv',
            )

# Read SDF file 
if files_option == 'MDL multiple SD file (*.sdf)':
    uploaded_file = st.file_uploader("Choose a SDF file")
    if uploaded_file is not None:
        st.header('CHEMICAL STRUCTURE VALIDATION AND STANDARDIZATION:')
        supplier = Chem.ForwardSDMolSupplier(uploaded_file,sanitize=False)
        failed_mols = []
        all_mols =[]
        wrong_structure=[]
        wrong_smiles=[]
        bad_index=[]
        for i, m in enumerate(supplier):
            structure = Chem.Mol(m)
            all_mols.append(structure)
            try:
                Chem.SanitizeMol(structure)
            except:
                failed_mols.append(m)
                wrong_smiles.append(Chem.MolToSmiles(m))
                wrong_structure.append(str(i+1))
                bad_index.append(i)

        
        st.write('Original data: ', len(all_mols), 'molecules')
        # st.write('Kept data: ', len(moldf), 'molecules')
        st.write('Failed data: ', len(failed_mols), 'molecules')
        if len(failed_mols)!=0:
            number =[]
            for i in range(len(failed_mols)):
                number.append(str(i+1))
            
            
            bad_molecules = pd.DataFrame({'No. failed molecule in original set': wrong_structure, 'SMILES of wrong structure: ': wrong_smiles, 'No.': number}, index=None)
            bad_molecules = bad_molecules.set_index('No.')
            st.dataframe(bad_molecules)

        # Standardization SDF file
        all_mols[:] = [x for i,x in enumerate(all_mols) if i not in bad_index] 
        records = []
        for i in range(len(all_mols)):
            record = Chem.MolToSmiles(all_mols[i])
            records.append(record)
        
        moldf_n = []
        for i,record in enumerate(records):
            canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(record),isomericSmiles = False)
            standard_record = standardize_smiles(canon_smi)
            m = Chem.MolFromSmiles(standard_record)
            moldf_n.append(m)
        
        st.write('Kept data: ', len(moldf_n), 'molecules')        
        if st.button('Run predictions!'):         
            structures=[]
            for i in moldf_n:
                m = Chem.MolToSmiles(i)
                structures.append(m)
            exp=[]
            std=[]
            chembl_id=[]
            y_pred_con=[]
            cpd_AD_vs=[]
            number =[]
            count=0
            struct=[]
            exp_tox=[]
            cas_id=[]
            y_pred_con_tox=[]
            cpd_AD_vs_tox=[]

            for i in structures:
                i=standardize_smiles(i)
                m = Chem.MolFromSmiles(i)
                inchi = str(Chem.MolToInchi(m))
                struct.append(i)
                if inchi in res:                    
                    exp.append(round((res[inchi][0]['pchembl_value_mean']), 2))
                    std.append(round((res[inchi][0]['pchembl_value_std']), 3))
                    chembl_id.append(str(res[inchi][0]['molecule_chembl_id']))
                    y_pred_con.append('see experimental value')
                    cpd_AD_vs.append('-')
                    count+=1         
                    number.append(count)
                                
                else:                   
                    # Calculate molecular descriptors
                    f_vs = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024, useFeatures=False, useChirality=False)]
                    X = rdkit_numpy_convert(f_vs)
                    #Predict activity                    
                    prediction_RF = load_model_GBR.predict(X)
                    prediction_SVM = load_model_SVM.predict(X)
                    y_pred=(prediction_RF+prediction_SVM)/2                                                         
                    y_pred_con.append(round(float(y_pred),3))
                    # Estimination AD
                    neighbors_k_vs = pairwise_distances(x_tr, Y=X, n_jobs=-1)
                    neighbors_k_vs.sort(0)
                    similarity_vs = neighbors_k_vs
                    cpd_value_vs = similarity_vs[0, :]
                    cpd_AD = np.where(cpd_value_vs <= model_AD_limit, "Inside AD", "Outside AD")
                    str_a = ''.join(cpd_AD)                    
                    cpd_AD_vs.append(str_a)
                    exp.append('-')
                    std.append('-')
                    chembl_id.append('-')
                    count+=1         
                    number.append(count)

                # search experimental toxicity value
                if inchi in res_tox:
                    exp_tox.append(str(res_tox[inchi][0]['TOX_VALUE']))
                    cas_id.append(str(res_tox[inchi][0]['CAS_Number']))
                    y_pred_con_tox.append('see experimental value')
                    cpd_AD_vs_tox.append('-')
                    
                else:
                    m_t = Chem.MolFromSmiles(i)
                    # Calculate molecular descriptors
                    f_vs_tox = [AllChem.GetMorganFingerprintAsBitVect(m_t, radius=2, nBits=1024, useFeatures=False, useChirality=False)]
                    X_tox = rdkit_numpy_convert(f_vs_tox)
                    # Estimination AD for toxicity
                    neighbors_k_vs_tox = pairwise_distances(x_tr_tox, Y=f_vs_tox, n_jobs=-1)
                    neighbors_k_vs_tox.sort(0)
                    similarity_vs_tox = neighbors_k_vs_tox
                    cpd_value_vs_tox = similarity_vs_tox[0, :]
                    cpd_AD_vs_tox_r = np.where(cpd_value_vs_tox <= model_AD_limit_tox, "Inside AD", "Outside AD")
                    prediction_SVM_tox = load_model_SVM_tox.predict(X_tox)
                    prediction_GBR_tox = load_model_GBR_tox.predict(X_tox)
                    y_pred_tox=(prediction_SVM_tox+prediction_GBR_tox)/2
                    
                    MolWt=ExactMolWt(Chem.MolFromSmiles(i))
                    value_ped_tox=(10**(y_pred_tox*-1)*1000)*MolWt
                    value_ped_tox=round(value_ped_tox[0], 4)
                    y_pred_con_tox.append(value_ped_tox)
                    cpd_AD_vs_tox.append(cpd_AD_vs_tox_r[0])
                    exp_tox.append("-")
                    cas_id.append("not detected")  

            common_inf = pd.DataFrame({'SMILES':struct, 'Predicted value, pIC50': y_pred_con, 'Applicability domain': cpd_AD_vs,
            'Experimental value, pIC50': exp,'Standard deviation': std,
            'Chemble ID': chembl_id,'No.': number,            
            'Predicted value toxicity, Ld50, mg/kg': y_pred_con_tox,
            'Applicability domain_tox': cpd_AD_vs_tox,
            'Experimental value toxicity, Ld50': exp_tox,
            'CAS number': cas_id}, index=None)
            predictions_pred = common_inf.set_index('No.')
            predictions_pred=predictions_pred.astype(str)
                        
        
            st.dataframe(predictions_pred)

                    

            def convert_df(df):
                    return df.to_csv().encode('utf-8')  
            csv = convert_df(predictions_pred)

            st.download_button(
                    label="Download results of prediction as CSV",
                    data=csv,
                    file_name='Results.csv',
                    mime='text/csv',
                )               