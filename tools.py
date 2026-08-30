import json
import re
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from anthropic import Anthropic

client = Anthropic()


metric_definitions = {
    "revenue": lambda data: data.loc["Total Revenue"],
    "gross_margin": lambda data: data.loc["Gross Profit"] / data.loc["Total Revenue"] * 100,
    "operating_margin": lambda data: data.loc["Operating Income"] / data.loc["Total Revenue"] * 100,
}


def get_financial_data(ticker, metrics, periods=8):
    stock = yf.Ticker(ticker)
    financials = stock.quarterly_financials

    results = {}
    not_found = []

    for metric in metrics:
        if metric not in metric_definitions:
            not_found.append(metric)
            continue
        try:
            series = metric_definitions[metric](financials.iloc[:, :periods])
            series.index = series.index.strftime("%Y-%m-%d")
            results[metric] = series.to_dict()
        except Exception:
            not_found.append(metric)

    return {
        "ticker": ticker,
        "data": results,
        "not_found": not_found,
        "periods_requested": periods,
        "periods_available": len(financials.columns),
        "source": "Yahoo Finance",
    }


def get_news(ticker, topic=None, max_articles=10):
    stock = yf.Ticker(ticker)
    articles = stock.news[:max_articles]

    if not articles:
        return {"ticker": ticker, "articles": [], "note": f"No recent news found for {ticker}"}

    if not topic:
        return {"ticker": ticker, "articles": articles, "note": None}

    numbered_list = "\n".join(
        f"{i+1}. Title: {a.get('content', {}).get('title', '')} | "
        f"Summary: {a.get('content', {}).get('summary', '')}"
        for i, a in enumerate(articles)
    )

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=50,
        system=f"You are filtering news articles for relevance to the topic '{topic}'. "
               f"Respond with ONLY a JSON array of the relevant article numbers, e.g. [1,3,4]. "
               f"If none are relevant, respond with [].",
        messages=[{"role": "user", "content": numbered_list}]
    )

    try:
        relevant_indices = json.loads(response.content[0].text)
        relevant_articles = [articles[i - 1] for i in relevant_indices]
    except Exception:
        relevant_articles = []

    if not relevant_articles:
        return {"ticker": ticker, "articles": [], "note": f"No relevant news found for {ticker} on topic '{topic}'"}

    return {"ticker": ticker, "articles": relevant_articles, "note": None}



section_patterns = {
    "risk factors": (r"item\s*1a\.?", r"item\s*1b\.?"),
    "mdna": (r"item\s*7\.?", r"item\s*7a\.?"),
}

HEADERS = {"User-Agent": "Simran Parikh sparikh88@gatech.edu"}


def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def _get_cik(ticker):
    resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=15)
    data = resp.json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    return None


def _get_latest_10k_url(cik):
    resp = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=HEADERS, timeout=15)
    filings = resp.json()["filings"]["recent"]
    for i, form in enumerate(filings["form"]):
        if form == "10-K":
            accession = filings["accessionNumber"][i].replace("-", "")
            doc = filings["primaryDocument"][i]
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
    return None


def _find_section(text, pattern):
    matches = [m.start() for m in re.finditer(pattern, text, re.IGNORECASE)]
    if len(matches) < 2:
        return None
    return matches[1]


def get_filing_section(ticker, section):
    section_key = section.lower()
    if section_key not in section_patterns:
        return {"error": f"Section '{section}' not recognized. Available: {list(section_patterns.keys())}"}

    cik = _get_cik(ticker)
    if not cik:
        return {"error": f"No CIK found for ticker '{ticker}'"}

    filing_url = _get_latest_10k_url(cik)
    if not filing_url:
        return {"error": f"No recent 10-K found for {ticker}"}

    filing_text = requests.get(filing_url, headers=HEADERS, timeout=15).text

    start_pattern, end_pattern = section_patterns[section_key]
    start = _find_section(filing_text, start_pattern)
    end = _find_section(filing_text, end_pattern)

    if start is None or end is None:
        return {"error": f"Section '{section}' not found in {ticker}'s latest 10-K"}

    return {
        "ticker": ticker,
        "section": section,
        "text": clean_html(filing_text[start:end]),
        "source_url": filing_url,
    }