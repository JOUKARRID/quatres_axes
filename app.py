import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import gdown
import os
import tempfile

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================

st.set_page_config(
    page_title="Offre Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLE PERSONNALISE
# ============================================================

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-header {
        padding: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #1f2937;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #6b7280;
        font-size: 1rem;
        margin: 0;
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #111827;
        border-left: 4px solid #1f2937;
        padding-left: 0.6rem;
        margin: 1.2rem 0 0.8rem 0;
    }
    
    .result-card {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .result-card .bdc-id {
        font-weight: 700;
        color: #111827;
        font-size: 1rem;
    }
    .result-card .bdc-similarite {
        color: #2563eb;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .result-card .bdc-label {
        color: #6b7280;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
        margin-bottom: 0.1rem;
    }
    .result-card .bdc-valeur {
        color: #111827;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    .result-card .bdc-articles {
        background-color: #f3f4f6;
        padding: 0.5rem 0.8rem;
        border-radius: 4px;
        margin-top: 0.3rem;
        font-size: 0.85rem;
        color: #374151;
    }
    .result-card .separator {
        border-top: 1px solid #e5e7eb;
        margin: 0.5rem 0;
    }
    
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-low { background-color: #dcfce7; color: #166534; }
    .badge-mid { background-color: #fef9c3; color: #854d0e; }
    .badge-high { background-color: #fee2e2; color: #991b1b; }
    
    .supplier-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        border-bottom: 1px solid #f3f4f6;
    }
    .supplier-rank {
        font-weight: 700;
        color: #111827;
        width: 1.8rem;
    }
    .supplier-name {
        flex: 1;
        color: #111827;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .supplier-proba {
        color: #4b5563;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .article-tag {
        display: inline-block;
        background-color: #e5e7eb;
        padding: 0.15rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        color: #374151;
        margin: 0.1rem 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>Offre Detector</h1>
    <p>Analyse comparative des bons de commande — marches similaires, niveau de concurrence et fournisseur probable</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# FONCTIONS DE TELECHARGEMENT DEPUIS GOOGLE DRIVE
# ============================================================

# IDs des fichiers sur Google Drive
# Extrait des IDs depuis l'URL : https://drive.google.com/file/d/ID/view
FILES = {
    'vecteurs_complets.npy': '1a-9gAfdupm2ew7LhaRHTukrYm0GuTDKS',  # Dossier ID
    'ids_bdc.npy': '1a-9gAfdupm2ew7LhaRHTukrYm0GuTDKS',  # Dossier ID
    'BDC_texte.csv': '1a-9gAfdupm2ew7LhaRHTukrYm0GuTDKS',  # Dossier ID
    'BDC_numerique.csv': '1a-9gAfdupm2ew7LhaRHTukrYm0GuTDKS',
    'modele_concurrence_xgb.pkl': '1a-9gAfdupm2ew7LhaRHTukrYm0GuTDKS',
    'modele_fournisseur_xgb.pkl': '1a-9gAfdupm2ew7LhaRHTukrYm0GuTDKS',
    'label_encoder_fournisseur.pkl': '1a-9gAfdupm2ew7LhaRHTukrYm0GuTDKS',
    'scaler_normalisation.pkl': '1a-9gAfdupm2ew7LhaRHTukrYm0GuTDKS'
}

def download_file_from_drive(file_id, output_path):
    """Telecharge un fichier depuis Google Drive avec gdown"""
    url = f'https://drive.google.com/uc?id={file_id}'
    try:
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        st.error(f"Erreur lors du telechargement de {output_path}: {e}")
        return False

def load_from_drive():
    """Charge tous les fichiers depuis Google Drive dans un dossier temporaire"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        st.info("Telechargement des fichiers depuis Google Drive...")
        
        # Telecharger les fichiers
        for filename in FILES.keys():
            file_path = os.path.join(tmp_dir, filename)
            if not download_file_from_drive(FILES[filename], file_path):
                return None, None, None, None, None, None, None, None
        
        st.info("Chargement des fichiers...")
        
        # Charger les fichiers
        try:
            # Vecteurs
            features = np.load(os.path.join(tmp_dir, 'vecteurs_complets.npy')).astype(np.float32)
            ids = np.load(os.path.join(tmp_dir, 'ids_bdc.npy'))
            
            # DataFrames
            df_texte = pd.read_csv(os.path.join(tmp_dir, 'BDC_texte.csv'))
            df_numerique = pd.read_csv(os.path.join(tmp_dir, 'BDC_numerique.csv'))
            
            # Modeles
            model_concurrence = joblib.load(os.path.join(tmp_dir, 'modele_concurrence_xgb.pkl'))
            model_fournisseur = joblib.load(os.path.join(tmp_dir, 'modele_fournisseur_xgb.pkl'))
            label_encoder = joblib.load(os.path.join(tmp_dir, 'label_encoder_fournisseur.pkl'))
            scaler = joblib.load(os.path.join(tmp_dir, 'scaler_normalisation.pkl'))
            
            # Fusionner les DataFrames
            df_complet = df_texte.merge(df_numerique, on='bdc_id', how='inner')
            
            st.success("Tous les fichiers ont ete charges avec succes !")
            
            return features, ids, df_complet, model_concurrence, model_fournisseur, label_encoder, scaler, df_texte, df_numerique
            
        except Exception as e:
            st.error(f"Erreur lors du chargement des fichiers: {e}")
            return None, None, None, None, None, None, None, None, None

# ============================================================
# CHARGEMENT DES MODELES
# ============================================================

@st.cache_resource
def load_models():
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    features, ids, df_complet, model_concurrence, model_fournisseur, label_encoder, scaler, df_texte, df_numerique = load_from_drive()
    
    if features is None:
        st.error("Impossible de charger les fichiers. Verifiez votre connexion.")
        return None
    
    return {
        'embedder': embedder,
        'features': features,
        'ids': ids,
        'df_complet': df_complet,
        'df_texte': df_texte,
        'df_numerique': df_numerique,
        'model_concurrence': model_concurrence,
        'model_fournisseur': model_fournisseur,
        'label_encoder': label_encoder,
        'scaler': scaler
    }

# ============================================================
# FONCTIONS UTILES
# ============================================================

def clean_value(val):
    if val is None:
        return "Non renseigne"
    if isinstance(val, float) and pd.isna(val):
        return "Non renseigne"
    if isinstance(val, str):
        if val.strip() in ['', 'nan', 'NaN', 'None', 'N/A', 'NA', 'null', 'Null', 'Inconnu']:
            return "Non renseigne"
        return val
    return val

# ============================================================
# FONCTIONS DE PREDICTION
# ============================================================

def nettoyer_texte(texte):
    texte = str(texte)
    texte = re.sub(r'[^a-zA-Z0-9\s\.\,\-\'\"]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte)
    return texte.strip()

def construire_document_bdc(objet, acheteur, categorie, lieu_execution, articles):
    lieu = lieu_execution if lieu_execution else "MAROC"
    
    objet_clean = nettoyer_texte(objet)
    acheteur_clean = nettoyer_texte(acheteur)
    categorie_clean = nettoyer_texte(categorie)
    lieu_clean = nettoyer_texte(lieu)
    
    doc = f"Objet : {objet_clean} Acheteur : {acheteur_clean} Categorie : {categorie_clean} Lieu : {lieu_clean}"
    
    if articles:
        doc += " [SEP] ARTICLES : "
        for a in articles:
            titre = nettoyer_texte(a.get('titre', ''))
            carac = nettoyer_texte(a.get('caracteristiques', ''))
            qte = a.get('quantite', 1)
            doc += f"{titre} {carac} quantite {qte} "
    
    return doc

def generer_embedding(data, document):
    embedder = data['embedder']
    embedding = embedder.encode([document], convert_to_numpy=True)[0]
    return embedding

def preparer_vecteur_complet(data, embedding):
    features = data['features']
    numeriques_moyennes = np.mean(features[:, 384:], axis=0)
    vecteur = np.concatenate([embedding, numeriques_moyennes])
    return vecteur.reshape(1, -1)

def trouver_marches_similaires(data, bdc_info, top_k=5):
    document = construire_document_bdc(**bdc_info)
    embedding = generer_embedding(data, document)
    vecteur = preparer_vecteur_complet(data, embedding)
    
    features = data['features']
    similarites = cosine_similarity(vecteur, features)[0]
    
    indices = np.argsort(similarites)[-top_k:][::-1]
    
    resultats = []
    df_complet = data['df_complet']
    
    for idx in indices:
        bdc_id = int(data['ids'][idx])
        ligne = df_complet[df_complet['bdc_id'] == bdc_id]
        
        if len(ligne) > 0:
            ligne = ligne.iloc[0]
            resultats.append({
                'bdc_id': bdc_id,
                'similarite': float(similarites[idx] * 100),
                'objet': clean_value(ligne.get('objet_clean', 'Non renseigne')),
                'titre_articles': clean_value(ligne.get('titre_clean', 'Non renseigne')),
                'caracteristiques': clean_value(ligne.get('caracteristiques_clean', 'Non renseigne')),
                'montant_ttc': ligne.get('montant_ttc', None),
                'nombre_devis': ligne.get('nombre_devis', None),
                'entreprise_attributaire': clean_value(ligne.get('entreprise_attributaire', 'Non renseigne')),
                'acheteur': clean_value(ligne.get('acheteur', 'Non renseigne')),
                'categorie': clean_value(ligne.get('categorie', 'Non renseigne')),
                'lieu_execution': clean_value(ligne.get('lieu_execution', 'Non renseigne'))
            })
    
    return resultats

def predire_concurrence(data, bdc_info):
    document = construire_document_bdc(**bdc_info)
    embedding = generer_embedding(data, document)
    vecteur = preparer_vecteur_complet(data, embedding)
    
    scaler = data['scaler']
    embedding_part = vecteur[:, :384]
    numeriques_part = vecteur[:, 384:]
    numeriques_scaled = scaler.transform(numeriques_part)
    vecteur_final = np.concatenate([embedding_part, numeriques_scaled], axis=1)
    
    pred = data['model_concurrence'].predict(vecteur_final)[0]
    probas = data['model_concurrence'].predict_proba(vecteur_final)[0]
    
    classes = ["Faible (0-10)", "Moyen (11-25)", "Forte (26+)"]
    return {
        'niveau': classes[pred],
        'probabilite': float(probas[pred] * 100),
        'classe': pred
    }

def predire_fournisseur(data, bdc_info, top_k=5):
    document = construire_document_bdc(**bdc_info)
    embedding = generer_embedding(data, document)
    vecteur = preparer_vecteur_complet(data, embedding)
    
    scaler = data['scaler']
    embedding_part = vecteur[:, :384]
    numeriques_part = vecteur[:, 384:]
    numeriques_scaled = scaler.transform(numeriques_part)
    vecteur_final = np.concatenate([embedding_part, numeriques_scaled], axis=1)
    
    probas = data['model_fournisseur'].predict_proba(vecteur_final)[0]
    indices_tries = np.argsort(probas)[::-1]
    
    resultats = []
    for idx in indices_tries:
        if probas[idx] > 0.01:
            nom = data['label_encoder'].inverse_transform([idx])[0]
            if nom and str(nom).strip() not in ['', 'nan', 'Inconnu', 'None']:
                resultats.append({
                    'nom': str(nom),
                    'probabilite': float(probas[idx] * 100)
                })
        if len(resultats) >= top_k:
            break
    
    return resultats

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### Configuration")
    st.caption("Renseignez tous les champs pour analyser un BDC.")
    
    st.markdown("---")
    st.markdown("**Axes d'analyse**")
    st.markdown("- Marches similaires (5)")
    st.markdown("- Niveau de concurrence")
    st.markdown("- Fournisseur probable")
    
    st.markdown("---")
    st.caption("Offre Detector - v1.0")

# ============================================================
# CHARGEMENT DES MODELES
# ============================================================

with st.spinner("Chargement des modeles..."):
    data = load_models()

if data is None:
    st.stop()

st.success("Modeles charges")

# ============================================================
# FORMULAIRE DE SAISIE
# ============================================================

st.markdown('<div class="section-title">Saisie du BDC</div>', unsafe_allow_html=True)

with st.form("bdc_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    
    with col1:
        objet = st.text_area(
            "Objet du BDC",
            height=80,
            placeholder="Ex : Achat de materiel informatique pour datacenter"
        )
        acheteur = st.text_input(
            "Acheteur",
            placeholder="Ex : MINISTERE DE LA TRANSITION NUMERIQUE"
        )
    
    with col2:
        categorie = st.selectbox(
            "Categorie",
            ["Fournitures", "Services", "Travaux", "Autre"]
        )
        lieu_execution = st.text_input(
            "Lieu d'execution",
            placeholder="Ex : RABAT, CASABLANCA"
        )
    
    st.markdown('<div class="section-title">Articles</div>', unsafe_allow_html=True)
    
    col_a1, col_a2, col_a3 = st.columns([2, 2, 1])
    with col_a1:
        titre_article = st.text_input("Titre de l'article", key="titre_article", placeholder="Ex : Ordinateur portable")
    with col_a2:
        carac_article = st.text_area("Caracteristiques", height=50, key="carac_article", placeholder="Ex : 16GB RAM, 512GB SSD")
    with col_a3:
        qte_article = st.number_input("Quantite", min_value=1, value=1, step=1, key="qte_article")
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        add_article = st.form_submit_button("Ajouter l'article")
        if add_article and titre_article and carac_article:
            if 'articles' not in st.session_state:
                st.session_state.articles = []
            st.session_state.articles.append({
                'titre': titre_article,
                'caracteristiques': carac_article,
                'quantite': qte_article
            })
            st.success("Article ajoute")
    
    with col_btn2:
        remove_article = st.form_submit_button("Supprimer le dernier")
        if remove_article and 'articles' in st.session_state and st.session_state.articles:
            st.session_state.articles.pop()
            st.info("Dernier article supprime")
    
    if 'articles' in st.session_state and st.session_state.articles:
        st.write("**Articles ajoutes :**")
        for i, a in enumerate(st.session_state.articles, 1):
            st.markdown(f"""
            <span class="article-tag">#{i}</span>
            <strong>{a['titre']}</strong>
            <span style="color:#6b7280;font-size:0.85rem;"> — {a['caracteristiques'][:60]}...</span>
            <span style="color:#6b7280;font-size:0.85rem;">x{a['quantite']}</span>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    submitted = st.form_submit_button("Analyser le BDC", use_container_width=True, type="primary")

# ============================================================
# TRAITEMENT DES RESULTATS
# ============================================================

if submitted:
    if not objet or not acheteur:
        st.error("Veuillez remplir l'objet et l'acheteur")
    elif 'articles' not in st.session_state or not st.session_state.articles:
        st.error("Veuillez ajouter au moins un article")
    else:
        bdc_info = {
            'objet': objet,
            'acheteur': acheteur,
            'categorie': categorie,
            'lieu_execution': lieu_execution,
            'articles': st.session_state.articles
        }
        
        with st.spinner("Analyse en cours..."):
            similaires = trouver_marches_similaires(data, bdc_info, top_k=5)
            concurrence = predire_concurrence(data, bdc_info)
            fournisseurs = predire_fournisseur(data, bdc_info, top_k=5)
        
        st.success("Analyse terminee")
        
        # RESUME
        st.markdown('<div class="section-title">Resume</div>', unsafe_allow_html=True)
        
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Marches similaires", f"{len(similaires)} trouves")
        with col_r2:
            badge_class = {
                "Faible (0-10)": "badge-low",
                "Moyen (11-25)": "badge-mid",
                "Forte (26+)": "badge-high"
            }.get(concurrence['niveau'], "badge-mid")
            st.markdown(f"""
            <div>
                <span style="font-size:0.8rem;color:#6b7280;">Niveau de concurrence</span><br>
                <span class="badge {badge_class}" style="font-size:0.9rem;">{concurrence['niveau']}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_r3:
            fournisseur_principal = fournisseurs[0]['nom'] if fournisseurs else "Indetermine"
            st.metric("Fournisseur probable", fournisseur_principal[:25] + "..." if len(fournisseur_principal) > 25 else fournisseur_principal)
        
        # MARCHES SIMILAIRES
        st.markdown('<div class="section-title">Marches similaires</div>', unsafe_allow_html=True)
        st.caption(f"Top {len(similaires)} marches les plus proches (similarite basee sur les articles)")
        
        for s in similaires:
            montant_str = f"{s['montant_ttc']:,.0f} MAD" if s['montant_ttc'] else "Non renseigne"
            devis_str = f"{s['nombre_devis']}" if s['nombre_devis'] else "Non renseigne"
            
            caracteristiques = s['caracteristiques']
            if len(caracteristiques) > 300:
                caracteristiques = caracteristiques[:300] + "..."
            
            st.markdown(f"""
            <div class="result-card">
                <div>
                    <span class="bdc-id">BDC {s['bdc_id']}</span>
                    <span class="bdc-similarite" style="float:right;">Similarite : {s['similarite']:.1f}%</span>
                </div>
                <div class="separator"></div>
                <div class="bdc-label">Objet</div>
                <div class="bdc-valeur">{s['objet']}</div>
                <div class="bdc-label">Articles</div>
                <div class="bdc-articles">{s['titre_articles']}</div>
                <div class="bdc-label">Caracteristiques</div>
                <div class="bdc-articles">{caracteristiques}</div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem;margin-top:0.5rem;">
                    <div>
                        <div class="bdc-label">Fournisseur</div>
                        <div class="bdc-valeur">{s['entreprise_attributaire']}</div>
                    </div>
                    <div>
                        <div class="bdc-label">Montant</div>
                        <div class="bdc-valeur">{montant_str}</div>
                    </div>
                    <div>
                        <div class="bdc-label">Nombre de devis</div>
                        <div class="bdc-valeur">{devis_str}</div>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.3rem;">
                    <div>
                        <div class="bdc-label">Acheteur</div>
                        <div class="bdc-valeur">{s['acheteur']}</div>
                    </div>
                    <div>
                        <div class="bdc-label">Lieu d'execution</div>
                        <div class="bdc-valeur">{s['lieu_execution']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # FOURNISSEURS PROBABLES
        st.markdown('<div class="section-title">Fournisseurs probables</div>', unsafe_allow_html=True)
        
        if fournisseurs:
            for i, f in enumerate(fournisseurs, 1):
                st.markdown(f"""
                <div class="supplier-row">
                    <span class="supplier-rank">#{i}</span>
                    <span class="supplier-name">{f['nom']}</span>
                    <span class="supplier-proba">{f['probabilite']:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(f['probabilite'] / 100)
        else:
            st.warning("Aucun fournisseur determine")
        
        st.markdown("---")
        st.caption("Les predictions sont basees sur l'historique des marches publics marocains.")