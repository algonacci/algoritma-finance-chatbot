import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import yfinance as yf
from typing import Dict, Union

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4-0125-preview")

@tool
def search_stock(symbol: str) -> Dict[str, Union[str, float, None]]:
    """
    Search for stock information using Yahoo Finance.
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1d")
        
        if len(hist) == 0:
            return {
                "error": f"No data found for {symbol}",
                "success": False
            }
        
        latest = hist.iloc[-1]
        
        return {
            "symbol": symbol,
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "description": info.get("longBusinessSummary"),
            "currency": info.get("currency"),
            "current_price": latest["Close"],
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "success": True
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }

# Create prompt template for stock analysis
template = """
Berdasarkan data saham berikut ini:

Nama Perusahaan: {name}
Sektor: {sector}
Industri: {industry}
Harga Saat Ini: {current_price} {currency}
Market Cap: {market_cap}
P/E Ratio: {pe_ratio}
Dividend Yield: {dividend_yield}

Deskripsi Bisnis:
{description}

Tolong berikan analisis komprehensif tentang perusahaan ini dalam Bahasa Indonesia yang mencakup:
1. Gambaran umum bisnis dan posisinya di industri
2. Kinerja keuangan berdasarkan metrik yang tersedia
3. Potensi dan risiko investasi

Analisis:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["name", "sector", "industry", "current_price", "currency", 
                    "market_cap", "pe_ratio", "dividend_yield", "description"]
)

def get_stock_data(ticker: str) -> dict:
    """Get and format stock data"""
    data = search_stock(ticker)
    return {
        "name": data.get("name", "N/A"),
        "sector": data.get("sector", "N/A"),
        "industry": data.get("industry", "N/A"),
        "current_price": data.get("current_price", "N/A"),
        "currency": data.get("currency", "N/A"),
        "market_cap": data.get("market_cap", "N/A"),
        "pe_ratio": data.get("pe_ratio", "N/A"),
        "dividend_yield": data.get("dividend_yield", "N/A"),
        "description": data.get("description", "N/A")
    }

# Create the chain using proper composition
chain = (
    RunnablePassthrough()
    | get_stock_data
    | prompt
    | llm
    | StrOutputParser()
)

def analyze_stock(ticker: str) -> str:
    """
    Analyze a stock by its ticker symbol.
    """
    try:
        result = chain.invoke(ticker)
        return result
    except Exception as e:
        return f"Error analyzing stock: {str(e)}"

if __name__ == "__main__":
    # Test with Indonesian stock
    ticker = str(input("Masukkan nama ticker saham: \n"))
    print(f"\nAnalisis untuk {ticker}:")
    analysis = analyze_stock(ticker)
    print(analysis)