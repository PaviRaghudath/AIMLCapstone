from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import chromadb.utils.embedding_functions as embedding_functions
import pandas as pd
import re
import sys
from datetime import datetime, timedelta

import os
import spacy
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.rag_engine import train_data, embedding_model, collection, corpus
nlp = spacy.load("en_core_web_sm")
from backend.routers.sales_prediction import predict_sales

# chroma_path = os.path.join(os.path.dirname(__file__), "..", "chromadb")

model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = PersistentClient(path="backend/chromadb")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="sales_data",
    embedding_function=sentence_transformer_ef
)

def spacy_intent_parser(fastApiRequest, query: str):
    doc = nlp(query.lower())
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    tokens = [token.text for token in doc]

    #  1. Highest-selling product family
    if "family" in nouns and ("sell" in verbs or "sale" in nouns or "most" in tokens or "top" in tokens):
        agg = train_data.groupby("family")["sales"].sum()
        top_family = agg.idxmax()
        value = agg.max()
        return f"Product family with highest total sales: **{top_family}** with **{value:,.2f}** units."

    #  2. Total sales for a specific store
    for token in doc:
        if token.text == "store" and token.nbor().like_num:
            store = int(token.nbor().text)
            if store in train_data["store_nbr"].unique():
                total = train_data[train_data["store_nbr"] == store]["sales"].sum()
                return f"Total sales for **Store {store}**: **{total:,.2f}** units."

    # 3. Total overall sales
    if "total" in tokens or "overall" in tokens and "sales" in tokens:
        total = train_data["sales"].sum()
        return f"Total sales across all stores and families: **{total:,.2f}** units."

    #  4. Holiday effect on sales
    if "holiday" in tokens and "sales" in tokens:
        if "is_holiday" in train_data.columns:
            avg_holiday = train_data[train_data["is_holiday"] == True]["sales"].mean()
            avg_non_holiday = train_data[train_data["is_holiday"] == False]["sales"].mean()
            return (f"Average sales during holidays: **{avg_holiday:,.2f}** units\n"
                    f"Average sales during non-holidays: **{avg_non_holiday:,.2f}** units")
        else:
            return "No holiday information available in the dataset."

    #  5. Promotion impact on sales
    if "promotion" in tokens or "promotions" in tokens:
        if "onpromotion" in train_data.columns:
            promo = train_data[train_data["onpromotion"] == 1]["sales"].mean()
            no_promo = train_data[train_data["onpromotion"] == 0]["sales"].mean()
            return (f"Average sales with promotions: **{promo:,.2f}** units\n"
                    f"Average sales without promotions: **{no_promo:,.2f}** units")
        else:
            return "Promotion data not available."

    #  6. Monthly sales trends
    if "month" in tokens and "sales" in tokens:
        if "date" in train_data.columns:
            train_data['month'] = pd.to_datetime(train_data['date']).dt.to_period('M')
            agg = train_data.groupby("month")["sales"].sum()
            return f"Monthly sales:\n{agg.round(2).to_string()}"

    #7. Prediction for tomorrow
   

    if "predict" in tokens or "prediction" in tokens and "tomorrow" in tokens:
        latest_date = pd.to_datetime(train_data["date"]).max()
        tomorrow = latest_date + timedelta(days=1)

        unique_stores = train_data["store_nbr"].unique()
        unique_families = train_data["family"].unique()

        future_data = pd.DataFrame([
            {"store_nbr": store, "family": fam, "onpromotion": 0, "date": tomorrow}
            for store in unique_stores
            for fam in unique_families
        ])

        try:
            pred_df = predict_sales(fastApiRequest, model_type="lgbm", input_df=future_data)
            top5 = pred_df.sort_values(by="predicted_sales", ascending=False).head(5)
            return "Top 5 predicted sales for tomorrow:\n" + top5[["store_nbr", "family", "predicted_sales"]].to_string(index=False)
        except Exception as e:
            return f"Prediction failed: {str(e)}"



    #  8. Top family in a store
    if "top" in tokens and "family" in tokens and "store" in tokens:
        for token in doc:
            if token.text == "store" and token.nbor().like_num:
                store = int(token.nbor().text)
                subset = train_data[train_data["store_nbr"] == store]
                if not subset.empty:
                    agg = subset.groupby("family")["sales"].sum()
                    top_family = agg.idxmax()
                    value = agg.max()
                    return f"In store {store}, top product family is **{top_family}** with **{value:,.2f}** units."
                else:
                    return f"Store {store} not found in data."

    #  9. Total sales for a specific product family
    for token in doc:
        if token.text == "family" and token.nbor().is_alpha:
            family = token.nbor().text.upper()
            if family in train_data["family"].unique():
                total = train_data[train_data["family"] == family]["sales"].sum()
                return f"Total sales for product family **{family}**: {total:,.2f} units."
            else:
                return f"Product family '{family}' not found in data."

    return None



def answer_question(fastApiRequest, query: str, top_k: int = 3) -> str:
    q = query.lower()

    # Rule 1: Product family with highest total sales
    if re.search(r"(which|what).*family.*(sold|has sold|had sold|most|highest)", q):
        agg = train_data.groupby("family")["sales"].sum()
        top_family = agg.idxmax()
        value = agg.max()
        return f"Product family with highest total sales: **{top_family}** with **{value:,.2f}** units."

    # Rule 2: Average sales per store
    if re.search(r"(average|mean).*sales.*store", q):
        agg = train_data.groupby("store_nbr")["sales"].mean()
        result = agg.round(2).to_string()
        return f"Average sales per store:\n{result}"

    # Rule 3: Total sales for specific product family
    match = re.search(r"sales.*for.*family\s+(\w+)", q)
    if match:
        family = match.group(1).upper()
        if family in train_data["family"].unique():
            total = train_data[train_data["family"] == family]["sales"].sum()
            return f"Total sales for product family **{family}**: {total:,.2f} units."
        else:
            return f"Product family '{family}' not found in data."

    # Rule 4: Store with highest total sales
    if re.search(r"(which|what).*store.*(highest|most).*sales", q):
        agg = train_data.groupby("store_nbr")["sales"].sum()
        top_store = agg.idxmax()
        value = agg.max()
        return f"Store with highest total sales: **Store {top_store}** with **{value:,.2f}** units."

    # Rule 5: Total overall sales
    if re.search(r"(total|overall).*sales", q):
        total = train_data["sales"].sum()
        return f"Total sales across all stores and families: **{total:,.2f}** units."

    # Rule 6: Monthly sales trends
    if re.search(r"sales by month", q) and "date" in train_data.columns:
        train_data['month'] = pd.to_datetime(train_data['date']).dt.to_period('M')
        agg = train_data.groupby("month")["sales"].sum()
        return f" Monthly sales:\n{agg.round(2).to_string()}"

    # Rule 7: Top family in specific store
    match = re.search(r"(top|highest).*family.*store\s+(\d+)", q)
    if match:
        store = int(match.group(2))
        subset = train_data[train_data["store_nbr"] == store]
        if not subset.empty:
            agg = subset.groupby("family")["sales"].sum()
            top_family = agg.idxmax()
            value = agg.max()
            return f" In store {store}, top product family is **{top_family}** with **{value:,.2f}** units."
        else:
            return f" Store {store} not found in data."

    # Rule 8: Holiday effect on sales
    if "holiday" in q and "sales" in q:
        if "is_holiday" in train_data.columns:
            avg_holiday = train_data[train_data["is_holiday"] == True]["sales"].mean()
            avg_non_holiday = train_data[train_data["is_holiday"] == False]["sales"].mean()
            return (f" Average sales during holidays: **{avg_holiday:,.2f}** units\n"
                    f" Average sales during non-holidays: **{avg_non_holiday:,.2f}** units")
        else:
            return " No holiday information available in the dataset."

    # Rule 9: Promotion impact on sales
    if "promotion" in q or "onpromotion" in q:
        if "onpromotion" in train_data.columns:
            promo = train_data[train_data["onpromotion"] == 1]["sales"].mean()
            no_promo = train_data[train_data["onpromotion"] == 0]["sales"].mean()
            return (f" Average sales with promotions: **{promo:,.2f}** units\n"
                    f" Average sales without promotions: **{no_promo:,.2f}** units")
        else:
            return " Promotion data not available."


    # Rule 11: Total sales for specific store
    match = re.search(r"(store\s+(\d+).*(sales|sell))|((sales|sell).*(store\s+(\d+)))", q)
    if match:
        store = int(match.group(2) or match.group(7))
        if store in train_data["store_nbr"].unique():
            total = train_data[train_data["store_nbr"] == store]["sales"].sum()
            return f"Total sales for **Store {store}**: **{total:,.2f}** units."
        else:
            return f"Store {store} not found in data."
        
    spacy_result = spacy_intent_parser(fastApiRequest, query)
    if spacy_result:
        return spacy_result
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        documents = results["documents"][0]
        scores = results["distances"][0]

        responses = [
            f"- {doc} (Similarity: {1 - score:.2f})"
            for doc, score in zip(documents, scores)
        ]

        return f" I couldn't parse that directly, but here are similar entries for: '{query}'\n" + "\n".join(responses)
    
    except Exception as e:
        return f" ChromaDB query failed: {str(e)}"





