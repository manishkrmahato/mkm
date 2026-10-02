import os
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler


file_path = r"Z:\HTML_Reference_Sample_Campus_Events.html"

with open(file_path, "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

events = soup.find_all("div", class_="event")

data = []

for event in events:
    name = event.find("h2")
    venue = event.find("p", class_="venue")
    date = event.find("p", class_="date")
    category = event.find("span", class_="category")
    fee = event.find("span", class_="fee")
    seats = event.find("p", class_="seats")

    data.append({
        "Event": name.get_text(strip=True) if name else "",
        "Venue": venue.get_text(strip=True) if venue else "",
        "Date": date.get_text(strip=True) if date else "",
        "Category": category.get_text(strip=True) if category else "",
        "Fee": fee.get_text(strip=True) if fee else "",
        "Seats": seats.get_text(strip=True) if seats else ""
    })

df = pd.DataFrame(data)


# url = "https://example.com/page.html"
# response = requests.get(url)
# response.raise_for_status()
# soup = BeautifulSoup(response.text, "html.parser")
# events = soup.find_all("div", class_="event")
#
# data = []
#
# for event in events:
#     name = event.find("h2")
#     venue = event.find("p", class_="venue")
#     date = event.find("p", class_="date")
#     category = event.find("span", class_="category")
#     fee = event.find("span", class_="fee")
#     seats = event.find("p", class_="seats")
#
#     data.append({
#         "Event": name.get_text(strip=True) if name else "",
#         "Venue": venue.get_text(strip=True) if venue else "",
#         "Date": date.get_text(strip=True) if date else "",
#         "Category": category.get_text(strip=True) if category else "",
#         "Fee": fee.get_text(strip=True) if fee else "",
#         "Seats": seats.get_text(strip=True) if seats else ""
#     })
#
# df = pd.DataFrame(data)

# OPTION 3: PDF TABLE

# import tabula
#
# pdf_path = r"Z:\question.pdf"
# tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
# df = pd.concat(tables, ignore_index=True)

# OPTION 4: PDF TEXT

# from PyPDF2 import PdfReader
#
# pdf_path = r"Z:\question.pdf"
# reader = PdfReader(pdf_path)
# text = ""
#
# for page in reader.pages:
#     text += page.extract_text()
#
# print(text)

# JSON

records = df.to_dict(orient="records")

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=4, ensure_ascii=False)

df = pd.read_json("data.json")

# EDA

print("FIRST 5 ROWS")
print(df.head())

print("\nSHAPE")
print(df.shape)

print("\nDATA TYPES")
print(df.dtypes)

print("\nSUMMARY")
print(df.describe(include="all"))

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATES")
print(df.duplicated().sum())

# DATA CLEANING

if "Venue" in df.columns:
    df["Venue"] = df["Venue"].replace("", pd.NA)
    df["Venue"] = df["Venue"].fillna("Not Specified")

df = df.drop_duplicates()

if "Fee" in df.columns:
    df["Fee"] = (
        df["Fee"]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("Free", "0")
        .astype(float)
    )

if "Seats" in df.columns:
    df["Seats"] = (
        df["Seats"]
        .astype(str)
        .str.replace("Seats:", "", regex=False)
        .str.strip()
        .astype(int)
    )

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# FEATURE ENGINEERING

if "Date" in df.columns:
    df["Day"] = df["Date"].dt.day_name()
    df["Month"] = df["Date"].dt.month_name()

if "Fee" in df.columns:
    df["Is_Free"] = df["Fee"] == 0

if "Seats" in df.columns:
    df["Seat_Category"] = pd.cut(
        df["Seats"],
        bins=[0, 49, 99, np.inf],
        labels=["Low", "Medium", "High"]
    )

# NORMALIZATION

if "Seats" in df.columns:
    scaler = MinMaxScaler()
    df["Seats_Normalized"] = scaler.fit_transform(df[["Seats"]])

print("\nCLEANED DATA")
print(df)

# ANALYSIS

if "Category" in df.columns:
    print("\nCATEGORY COUNTS")
    print(df["Category"].value_counts())

if "Fee" in df.columns:
    print("\nAVERAGE FEE")
    print(df["Fee"].mean())

if "Seats" in df.columns:
    print("\nAVERAGE SEATS")
    print(df["Seats"].mean())

if "Is_Free" in df.columns:
    print("\nFREE EVENTS")
    print(df[df["Is_Free"]]["Event"].tolist())

if "Seats" in df.columns and "Event" in df.columns:
    print("\nEVENT WITH MAXIMUM SEATS")
    print(df.loc[df["Seats"].idxmax(), ["Event", "Seats"]])

    print("\nEVENT WITH MINIMUM SEATS")
    print(df.loc[df["Seats"].idxmin(), ["Event", "Seats"]])

if "Category" in df.columns:
    print("\nCATEGORY WITH MOST EVENTS")
    print(df["Category"].value_counts().idxmax())

if "Category" in df.columns and "Seats" in df.columns:
    print("\nAVERAGE SEATS BY CATEGORY")
    print(df.groupby("Category")["Seats"].mean())

if "Fee" in df.columns and "Seats" in df.columns:
    print("\nCORRELATION")
    print(df[["Fee", "Seats"]].corr())

# VISUALIZATION

if "Fee" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Fee"], kde=True)
    plt.title("Distribution of Event Fees")
    plt.xlabel("Fee")
    plt.ylabel("Frequency")
    plt.show()

if "Category" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Category")
    plt.title("Number of Events by Category")
    plt.xlabel("Category")
    plt.ylabel("Number of Events")
    plt.xticks(rotation=30)
    plt.show()

if "Event" in df.columns and "Seats" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="Event", y="Seats")
    plt.title("Seats Available for Each Event")
    plt.xlabel("Event")
    plt.ylabel("Seats")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

if "Fee" in df.columns and "Seats" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="Fee",
        y="Seats",
        hue="Category" if "Category" in df.columns else None,
        s=100
    )
    plt.title("Fee vs Seats")
    plt.xlabel("Fee")
    plt.ylabel("Seats")
    plt.show()

if "Category" in df.columns and "Seats" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Category", y="Seats")
    plt.title("Seat Distribution by Category")
    plt.xlabel("Category")
    plt.ylabel("Seats")
    plt.xticks(rotation=30)
    plt.show()

if "Is_Free" in df.columns:
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x="Is_Free")
    plt.title("Free vs Paid Events")
    plt.xlabel("Is Free")
    plt.ylabel("Number of Events")
    plt.show()

if "Fee" in df.columns and "Seats" in df.columns:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        df[["Fee", "Seats"]].corr(),
        annot=True,
        cmap="coolwarm"
    )
    plt.title("Correlation Heatmap")
    plt.show()

# SAVE FINAL DATA

df.to_csv("cleaned_data.csv", index=False)

print("\nJSON saved as: data.json")
print("CSV saved as: cleaned_data.csv")