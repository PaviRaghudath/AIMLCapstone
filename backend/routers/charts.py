import pandas as pd
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from preprocessing.data_loader import load_all_data
from preprocessing.feature_engineering import prepare_features
from sklearn.preprocessing import LabelEncoder
from fastapi import APIRouter, Request
import pandas as pd
from collections import defaultdict
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


# data = load_all_data()
# df = data["train"]
# df = prepare_features(df, data["oil"], data["holidays"], data["transactions"], data["stores"])

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")


def total_sales_by_store(df):
    # print("Inside the total sales by store:",df[['store_nbr', 'sales']].head())
    # print("Inside the second one total sales by store:",df[['store_nbr', 'sales']].isnull().sum())
    # print("Inside the third one total sales by store:",df.shape)
    data = df.groupby('store_nbr')['sales'].sum().reset_index()
    # print("Inside the fourth one total sales by store:",data.head()) 
    
    fig = px.bar(data, x='store_nbr', y='sales', title='Total Sales by Store')
    fig.update_layout(
    yaxis=dict(tickformat=".2s"),  
    # bargap=0.2,
    # title=dict(
    #     text="Total Sales by Store",
    #     pad=dict(t=0, b=0)
    # ),
    # margin=dict(t=40, b=100),
    # title_font=dict(size=14),
    xaxis_tickangle=-45,
    # xaxis=dict(automargin=True)
    )
    return plot(fig, output_type='div', include_plotlyjs=False)

def quarterly_sales_comparison(df):
    df['quarter'] = pd.to_datetime(df['date']).dt.to_period('Q')
    df['year'] = pd.to_datetime(df['date']).dt.year
    data = df.groupby(['quarter', 'year'])['sales'].sum().reset_index()
    data['quarter'] = data['quarter'].astype(str)
    fig = px.bar(data, x='quarter', y='sales', color='year', barmode='group',
                 title='Quarterly Sales Comparison by Year')
    fig.update_layout(
    # title=dict(
    #     text="Quarterly Sales Comparison by Year",
    #     pad=dict(t=0, b=0)
    # ),
    # margin=dict(t=40, b=100),
    # title_font=dict(size=14),
    xaxis_tickangle=-45,
    # xaxis=dict(automargin=True)
    )
    return plot(fig, output_type='div', include_plotlyjs=False)

def average_sales_by_family(df):
    data = df.groupby('family')['sales'].mean().reset_index()
    fig = px.bar(data, x='family', y='sales', title='Average Sales by Product Family')
    fig.update_layout(
    # title=dict(
    #     text="Average Sales by Product Family",
    #     pad=dict(t=0, b=0)
    # ),
    # margin=dict(t=40, b=100),
    # title_font=dict(size=14),
    xaxis_tickangle=-45,
    # xaxis=dict(automargin=True)
    )
    return plot(fig, output_type='div', include_plotlyjs=False)

def monthly_sales_comparison(df):
    df['monthly'] = pd.to_datetime(df['date']).dt.to_period('M')
    df['year'] = pd.to_datetime(df['date']).dt.year
    data = df.groupby(['monthly', 'year'])['sales'].sum().reset_index()
    data['monthly'] = data['monthly'].astype(str)
    fig = px.bar(data, x='monthly', y='sales', color='year', barmode='group',
                 title='Monthly Sales Comparison by Year')
    return plot(fig, output_type='div', include_plotlyjs=False)

def sales_distribution_by_holiday_promo(df):
    df_copy = df.copy()
    df_copy['holiday_promo'] = df_copy.apply(
        lambda row: (
            'Holiday & Promo' if row['is_holiday'] == 1 and row['onpromotion'] > 0 else
            'Holiday & No Promo' if row['is_holiday'] == 1 else
            'Not Holiday & Promo' if row['onpromotion'] > 0 else
            'Not Holiday & No Promo'
        ),
        axis=1
    )
    category_order = [
        'Not Holiday & No Promo',
        'Not Holiday & Promo',
        'Holiday & No Promo',
        'Holiday & Promo'
    ]
    df_copy['holiday_promo'] = pd.Categorical(df_copy['holiday_promo'], categories=category_order, ordered=True)

    fig = px.box(
        df_copy,
        x='holiday_promo',
        y='sales',
        category_orders={'holiday_promo': category_order},
        title='Sales Distribution by Holiday and Promotion',
        labels={'holiday_promo': 'Holiday & Promotion Category', 'sales': 'Sales'},
        points='all',
        color='holiday_promo',
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(showlegend=False)
    return plot(fig, output_type='div', include_plotlyjs=False)

def state_sales(df):
    df['date'] = pd.to_datetime(df['date'])
    state_sales_total = df.groupby('state')['sales'].sum().reset_index()

    fig = px.pie(
        state_sales_total,
        names='state',
        values='sales',
        title='Total Sales by State (All Years)',
        hole=0.4 
    )
    fig.update_layout(
        height=400,
        margin=dict(t=50, r=150, b=50, l=50),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.1,
            font=dict(size=10)
        )
    )
    return plot(fig, output_type='div', include_plotlyjs=False)


@router.get("/", response_class=HTMLResponse)
def get_overview(request: Request):

    df = request.app.state.features_df
    df.dropna(inplace=True)

    total_sales = total_sales_by_store(df)
    quarterly_sales = quarterly_sales_comparison(df)
    avg_family_sales = average_sales_by_family(df)
    mon_sales = monthly_sales_comparison(df)
    holi_promo_sales = sales_distribution_by_holiday_promo(df)
    sta_sales = state_sales(df)

    return templates.TemplateResponse("index_v2.html", {
        "request": request,
        "total_sales_chart": total_sales,
        "quarterly_sales_chart": quarterly_sales,
        "family_sales_chart": avg_family_sales,
        "mon_sales":mon_sales,
        "holi_promo_sales":holi_promo_sales,
        "sta_sales":sta_sales

    })