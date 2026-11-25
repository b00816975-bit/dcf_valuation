
# app.py
# Company-specific defaults + five-year DCF.
# Libraries: streamlit, yfinance, pandas, numpy, plotly
#
# In a notebook, put %%writefile app.py on a separate first line
# ABOVE this code. Do not include notebook magic inside app.py.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ------------------------------------------------------------
# 1. Editable assumption tables
# ------------------------------------------------------------
# LAST UPDATED: 2026-11-24; review monthly
# Approximate 10-year government-yield PLACEHOLDERS, in percent.
# These are not verified live quotes. Replace them with your data.
# EUR uses a generic core-euro government-yield placeholder.
RISK_FREE_BY_CURRENCY = {
    "USD": 4.0,
    "EUR": 2.5,
    "GBP": 4.5,
    "CHF": 0.5,
    "JPY": 1.5,
    "SGD": 2.5,
    "HKD": 3.5,
    "AUD": 4.0,
    "CAD": 3.0,
    "CNY": 2.0,
    "INR": 6.5,
    "KRW": 3.0,
}

# Simplified modeling assumptions, in percent.
# These are not a comprehensive tax-law database.
STATUTORY_TAX_BY_COUNTRY = {
    "United States": 25.0,
    "France": 25.0,
    "Singapore": 17.0,
    "Japan": 30.0,
    "United Kingdom": 25.0,
    "Germany": 30.0,
    "Switzerland": 19.0,
    "Hong Kong": 16.5,
    "Australia": 30.0,
    "Canada": 26.5,
    "China": 25.0,
    "India": 25.0,
    "South Korea": 26.5,
    "Netherlands": 25.8,
    "Ireland": 12.5,
    "Italy": 28.0,
    "Spain": 25.0,
    "Sweden": 20.6,
    "Denmark": 22.0,
    "Norway": 22.0,
    "Finland": 20.0,
    "Belgium": 25.0,
    "Brazil": 34.0,
    "Mexico": 30.0,
    "Taiwan": 20.0,
}

COUNTRY_ALIASES = {
    "US": "United States",
    "USA": "United States",
    "UK": "United Kingdom",
    "Korea, Republic of": "South Korea",
    "Republic of Korea": "South Korea",
}

BETA_FLOOR_BY_SECTOR = {
    "Consumer Defensive": 0.6,
    "Utilities": 0.6,
    "Healthcare": 0.7,
}

SECTOR_ROIC_DEFAULTS = {
    "Technology": 25.0,
    "Consumer Defensive": 12.0,
    "Healthcare": 15.0,
    "Industrials": 12.0,
    "Energy": 9.0,
    "Utilities": 7.0,
    "Materials": 10.0,
    "Basic Materials": 10.0,
}

TERMINAL_GROWTH_BY_CURRENCY = {
    "USD": 2.0, "EUR": 2.0, "GBP": 2.0, "CHF": 2.0, "JPY": 2.0,
    "SGD": 2.5, "HKD": 2.5, "AUD": 2.5, "CAD": 2.5,
    "INR": 3.0, "CNY": 3.0,
}


# ------------------------------------------------------------
# 2. Page and helpers
# ------------------------------------------------------------
st.set_page_config(page_title="Simple DCF Valuation", layout="wide")
st.title("Simple DCF Company Valuation")
st.write(
    "Company-specific defaults, editable assumptions, a five-year DCF, "
    "reverse DCF, and valuation sensitivity."
)
st.caption(
    "Educational scenario-analysis tool, not investment advice. "
    "Defaults are starting points, not forecasts."
)


def safe_number(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def format_money(value, currency=""):
    number = safe_number(value)
    if number is None:
        return "Not available"
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:,.2f}B {currency}".strip()
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:,.2f}M {currency}".strip()
    return f"{number:,.2f} {currency}".strip()


def pct(value):
    return "Not available" if value is None else f"{value * 100:.2f}%"


def normalize_label(label):
    return "".join(c for c in str(label).lower() if c.isalnum())


def statement_row(frame, label):
    if frame.empty:
        return None
    wanted = normalize_label(label)
    for index in frame.index:
        if normalize_label(index) == wanted:
            row = frame.loc[index]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return pd.to_numeric(row, errors="coerce")
    return None


def row_value(frame, label, column):
    row = statement_row(frame, label)
    return None if row is None else safe_number(row.get(column))


def first_row_value(frame, labels, column):
    for label in labels:
        value = row_value(frame, label, column)
        if value is not None:
            return value, label
    return None, None


def clean_statement(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    dates = pd.to_datetime(frame.columns, errors="coerce")
    valid = ~pd.isna(dates)
    frame = frame.loc[:, valid]
    frame.columns = dates[valid]
    frame = frame.loc[:, ~frame.columns.duplicated()]
    return frame.sort_index(axis=1, ascending=False)


def get_statement(company, frequency, messages, balance=False):
    try:
        if balance:
            frame = company.get_balance_sheet(freq=frequency, pretty=True)
        else:
            frame = company.get_income_stmt(freq=frequency, pretty=True)
        return clean_statement(frame)
    except Exception as error:
        kind = "balance sheet" if balance else "income statement"
        messages.append(f"{frequency} {kind}: {error}")
        return pd.DataFrame()


def build_periods(trailing, quarterly, annual):
    periods = []

    if not trailing.empty:
        date = trailing.columns[0]
        periods.append({
            "frame": trailing.loc[:, [date]],
            "label": f"Yahoo TTM ending {date.date()}",
            "date": date,
            "priority": 0,
            "ttm": True,
        })

    if quarterly.shape[1] >= 4:
        four = quarterly.iloc[:, :4]
        dates = four.columns
        gaps = [(dates[i] - dates[i + 1]).days for i in range(3)]
        if all(60 <= gap <= 120 for gap in gaps):
            periods.append({
                "frame": four,
                "label": f"Four-quarter TTM ending {dates[0].date()}",
                "date": dates[0],
                "priority": 1,
                "ttm": True,
            })

    if not annual.empty:
        date = annual.columns[0]
        periods.append({
            "frame": annual.loc[:, [date]],
            "label": f"Latest annual ending {date.date()}",
            "date": date,
            "priority": 2,
            "ttm": False,
        })

    periods.sort(key=lambda p: (-p["date"].value, p["priority"]))
    return periods


def annual_period(frame, date):
    return {
        "frame": frame.loc[:, [date]],
        "label": f"Annual ending {date.date()}",
        "date": date,
        "ttm": False,
    }


def period_total(period, label):
    row = statement_row(period["frame"], label)
    if row is None:
        return None
    values = row.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    return float(values.sum())


def operating_income_for_period(period):
    for label in ["Operating Income", "EBIT"]:
        value = period_total(period, label)
        if value is not None:
            return value, label

    gross = period_total(period, "Gross Profit")
    expense = period_total(period, "Operating Expense")
    if gross is not None and expense is not None:
        return gross - expense, "Gross Profit minus Operating Expense"
    return None, None


def margin_for_period(period):
    income, method = operating_income_for_period(period)
    revenue = period_total(period, "Total Revenue")
    if income is None or revenue is None or revenue <= 0:
        return None, method
    return income / revenue, method


def effective_tax(period):
    provision = period_total(period, "Tax Provision")
    pretax = period_total(period, "Pretax Income")
    if provision is None or pretax is None or pretax <= 0:
        return None
    return safe_number(provision / pretax)


def reset_defaults():
    """Clear only valuation widgets; keep the loaded company."""
    for key in list(st.session_state.keys()):
        if key.startswith("dcf_"):
            del st.session_state[key]


def is_financial_company(sector, industry):
    text = f"{sector} {industry}".lower()
    return any(word in text for word in [
        "bank", "insurance", "reit",
        "real estate investment trust", "capital markets",
    ])


# ------------------------------------------------------------
# 3. Download data and build company defaults
# ------------------------------------------------------------
def load_company(ticker):
    company = yf.Ticker(ticker)
    messages = []
    warnings = []

    try:
        info = company.info or {}
    except Exception as error:
        info = {}
        messages.append(f"Company information: {error}")

    trailing = get_statement(company, "trailing", messages)
    quarterly = get_statement(company, "quarterly", messages)
    annual = get_statement(company, "yearly", messages)
    balance_annual = get_statement(company, "yearly", messages, balance=True)
    balance_quarterly = get_statement(
        company, "quarterly", messages, balance=True
    )
    periods = build_periods(trailing, quarterly, annual)

    sector = info.get("sector") or "Not available"
    country = COUNTRY_ALIASES.get(
        info.get("country"), info.get("country") or "Not available"
    )
    quote_currency = info.get("currency") or ""
    financial_currency = info.get("financialCurrency") or ""

    # Do not assume that trading currency equals statement currency.
    risk_free_default = RISK_FREE_BY_CURRENCY.get(financial_currency, 3.5)
    if financial_currency not in RISK_FREE_BY_CURRENCY:
        warnings.append(
            f"Financial-statement currency '{financial_currency or 'missing'}' "
            "is not in the risk-free table. Using the 3.5% fallback."
        )
    risk_free_source = (
        f"Editable 10-year-yield placeholder for {financial_currency}; "
        "table dated 2026-11-24, not a live quote"
        if financial_currency in RISK_FREE_BY_CURRENCY
        else "Unknown/missing statement currency: 3.5% fallback"
    )

    # Operating income and matching-period revenue.
    operating_income = None
    operating_source = "Not available"
    operating_method = None
    operating_period = None
    revenue = None
    revenue_source = "Not available"
    reported_margin = None

    for period in periods:
        value, method = operating_income_for_period(period)
        if value is not None:
            operating_income = value
            operating_method = method
            operating_period = period
            operating_source = f"{method} | {period['label']}"
            revenue = period_total(period, "Total Revenue")
            if revenue is not None:
                revenue_source = f"Total Revenue | {period['label']}"
                if revenue > 0:
                    reported_margin = value / revenue
            break

    if revenue is None:
        for period in periods:
            value = period_total(period, "Total Revenue")
            if value is not None:
                revenue = value
                revenue_source = f"Total Revenue | {period['label']}"
                break
    if revenue is None:
        revenue = safe_number(info.get("totalRevenue"))
        if revenue is not None:
            revenue_source = "Yahoo totalRevenue; reporting period unverified"

    # Beta: missing raw beta produces a 1.0 default.
    raw_beta = safe_number(info.get("beta"))
    blume_beta = 0.67 * raw_beta + 0.33 if raw_beta is not None else 1.0
    beta_floor = BETA_FLOOR_BY_SECTOR.get(sector, 0.8)
    beta_default = max(blume_beta, beta_floor)
    beta_source = (
        f"max(0.67 × {raw_beta:.4f} + 0.33, sector floor {beta_floor:.2f})"
        if raw_beta is not None
        else f"Missing raw beta: 1.0 fallback; sector floor {beta_floor:.2f}"
    )
    if raw_beta is None:
        warnings.append("Raw beta is missing. The beta default is 1.0.")

    # Use the most recent computable effective rate, not an older
    # conveniently low/high observation.
    raw_tax = None
    raw_tax_source = "Not available"
    for period in periods:
        candidate = effective_tax(period)
        if candidate is not None:
            raw_tax = candidate
            raw_tax_source = (
                f"Tax Provision / Pretax Income | {period['label']}"
            )
            break

    statutory = STATUTORY_TAX_BY_COUNTRY.get(country, 25.0)
    if raw_tax is not None and 0.10 <= raw_tax <= 0.35:
        tax_default = raw_tax * 100
        tax_source = raw_tax_source + "; within 10%–35%"
    else:
        tax_default = statutory
        tax_source = (
            f"Simplified statutory assumption for {country}"
            if country in STATUTORY_TAX_BY_COUNTRY
            else "Country not in table: 25% statutory fallback"
        )
        warnings.append(
            "Effective tax rate is missing or outside 10%–35%. "
            f"Using {tax_default:.2f}% from the statutory assumption table."
        )

    # Three-year CAGR requires four annual observations.
    raw_cagr = None
    growth_source = "Four annual revenue observations unavailable; manual entry required"
    if annual.shape[1] >= 4:
        dates = annual.columns[:4]
        gaps = [(dates[i] - dates[i + 1]).days for i in range(3)]
        newest = row_value(annual, "Total Revenue", dates[0])
        oldest = row_value(annual, "Total Revenue", dates[3])

        if (
            all(300 <= gap <= 430 for gap in gaps)
            and newest is not None and oldest is not None
            and newest > 0 and oldest > 0
        ):
            raw_cagr = (newest / oldest) ** (1 / 3) - 1
            growth_source = (
                f"Three-year CAGR: annual revenue {dates[3].date()} to "
                f"{dates[0].date()}, capped to -5%–15%"
            )

    growth_default = (
        float(np.clip(raw_cagr * 100, -5.0, 15.0))
        if raw_cagr is not None else None
    )
    if growth_default is None:
        warnings.append(
            "A three-year revenue CAGR cannot be calculated. "
            "Enter starting growth manually; no single-year fallback is used."
        )

    # Analyst revenue estimates are references only.
    analyst_rows = []
    try:
        estimates = company.revenue_estimate
        if isinstance(estimates, pd.DataFrame) and not estimates.empty:
            for period_name, estimate in estimates.iterrows():
                growth_estimate = None
                for field in ["growth", "revenueGrowth"]:
                    if field in estimate.index:
                        growth_estimate = safe_number(estimate[field])
                        if growth_estimate is not None:
                            break

                average = safe_number(estimate.get("avg"))
                year_ago = safe_number(estimate.get("yearAgoRevenue"))
                derived = False
                if (
                    growth_estimate is None and average is not None
                    and year_ago is not None and year_ago > 0
                ):
                    growth_estimate = average / year_ago - 1
                    derived = True

                if growth_estimate is not None:
                    analyst_rows.append({
                        "Period": str(period_name),
                        "Revenue growth": growth_estimate,
                        "Source": (
                            "Analyst average / year-ago revenue - 1"
                            if derived else "Yahoo analyst revenue estimate growth"
                        ),
                    })
    except Exception as error:
        messages.append(f"Analyst revenue estimates: {error}")

    # Operating margin: equally weight three annual margins + one TTM.
    margin_components = []
    for date in annual.columns[:3]:
        period = annual_period(annual, date)
        value, method = margin_for_period(period)
        margin_components.append({
            "Component": period["label"],
            "Margin": value,
            "Source": method or "Not available",
        })

    ttm_margin = None
    for period in periods:
        if period["ttm"]:
            value, method = margin_for_period(period)
            if value is not None:
                ttm_margin = value
                margin_components.append({
                    "Component": period["label"],
                    "Margin": value,
                    "Source": method,
                })
                break
    if ttm_margin is None:
        margin_components.append({
            "Component": "TTM",
            "Margin": None,
            "Source": "Not available",
        })

    available_margins = [
        item["Margin"] for item in margin_components
        if item["Margin"] is not None
    ]
    target_margin_default = (
        float(np.mean(available_margins)) * 100
        if available_margins else None
    )
    margin_source = (
        f"Equal-weight mean of {len(available_margins)} available "
        "components: up to 3 annual margins plus TTM"
        if available_margins else "No reliable margins; manual entry required"
    )
    if len(available_margins) < 4:
        warnings.append(
            f"Only {len(available_margins)} of 4 margin components are available. "
            "The margin default uses the available components only."
        )

    current_margin_default = (
        ttm_margin * 100 if ttm_margin is not None
        else reported_margin * 100 if reported_margin is not None else None
    )

    # ROIC: latest balance sheet, with matching annual/TTM income if possible.
    balance_candidates = [
        frame for frame in [balance_quarterly, balance_annual]
        if not frame.empty
    ]
    latest_balance = (
        max(balance_candidates, key=lambda frame: frame.columns[0])
        if balance_candidates else pd.DataFrame()
    )

    raw_roic = None
    roic_source = ""
    invested_capital = None

    if not latest_balance.empty:
        balance_date = latest_balance.columns[0]
        bs_debt, debt_label = first_row_value(
            latest_balance, ["Total Debt"], balance_date
        )
        if bs_debt is None:
            current_debt, current_label = first_row_value(
                latest_balance,
                ["Current Debt And Capital Lease Obligation", "Current Debt"],
                balance_date,
            )
            long_debt, long_label = first_row_value(
                latest_balance,
                ["Long Term Debt And Capital Lease Obligation", "Long Term Debt"],
                balance_date,
            )
            if current_debt is not None and long_debt is not None:
                bs_debt = current_debt + long_debt
                debt_label = f"{current_label} + {long_label}"

        equity, equity_label = first_row_value(
            latest_balance,
            ["Stockholders Equity", "Total Equity Gross Minority Interest"],
            balance_date,
        )
        cash, cash_label = first_row_value(
            latest_balance,
            ["Cash Cash Equivalents And Short Term Investments",
             "Cash And Cash Equivalents"],
            balance_date,
        )

        income_candidates = list(periods)
        for date in annual.columns:
            income_candidates.append(annual_period(annual, date))

        income_candidates.sort(
            key=lambda p: (
                abs((p["date"] - balance_date).days),
                0 if p.get("ttm") else 1,
            )
        )

        for period in income_candidates:
            # Avoid treating a stale annual figure as current ROIC.
            if abs((period["date"] - balance_date).days) > 120:
                continue
            income, method = operating_income_for_period(period)
            if income is None:
                continue

            if all(v is not None for v in [bs_debt, equity, cash]):
                invested_capital = bs_debt + equity - cash
                if invested_capital > 0:
                    nopat = income * (1 - tax_default / 100)
                    raw_roic = safe_number(nopat / invested_capital)
                    roic_source = (
                        f"NOPAT using {method}, {period['label']}, "
                        f"tax default {tax_default:.2f}%; divided by "
                        f"({debt_label} + {equity_label} - {cash_label}) "
                        f"at {balance_date.date()}; raw result clipped to 8%–40%"
                    )
            break

    if raw_roic is not None:
        roic_default = float(np.clip(raw_roic * 100, 8.0, 40.0))
    else:
        roic_default = SECTOR_ROIC_DEFAULTS.get(sector, 12.0)
        roic_source = (
            f"Sector fallback for {sector}: {roic_default:.2f}%; "
            "fallback is not subject to historical-ROIC clipping"
        )
        warnings.append(
            "ROIC could not be computed from usable income and invested "
            f"capital. Using the {roic_default:.2f}% sector fallback."
        )

    terminal_base = TERMINAL_GROWTH_BY_CURRENCY.get(financial_currency, 2.0)
    terminal_default = min(terminal_base, risk_free_default - 0.5)
    terminal_source = (
        f"Currency rule {terminal_base:.2f}%; capped at risk-free default "
        "minus 0.50 percentage points"
    )

    total_debt = safe_number(info.get("totalDebt"))
    total_cash = safe_number(info.get("totalCash"))
    market_cap = safe_number(info.get("marketCap"))

    if (
        total_debt is not None and market_cap is not None
        and total_debt >= 0 and market_cap > 0
        and financial_currency and financial_currency == quote_currency
    ):
        debt_weight = total_debt / (total_debt + market_cap) * 100
        debt_weight_source = "Debt / (debt + market cap), matching currencies"
    else:
        debt_weight = 20.0
        debt_weight_source = "Illustrative 20% fallback: incomplete/currency-mismatched data"

    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "currency": quote_currency,
        "financial_currency": financial_currency,
        "country": country,
        "sector": sector,
        "industry": info.get("industry") or "Not available",
        "price": safe_number(info.get("currentPrice")),
        "shares": safe_number(info.get("sharesOutstanding")),
        "market_cap": market_cap,
        "debt": total_debt,
        "cash": total_cash,
        "revenue": revenue,
        "revenue_source": revenue_source,
        "operating_income": operating_income,
        "operating_source": operating_source,
        "operating_method": operating_method,
        "raw_tax": raw_tax,
        "raw_tax_source": raw_tax_source,
        "raw_beta": raw_beta,
        "blume_beta": blume_beta,
        "beta_floor": beta_floor,
        "raw_cagr": raw_cagr,
        "analyst_rows": analyst_rows,
        "margin_components": margin_components,
        "raw_roic": raw_roic,
        "invested_capital": invested_capital,
        "warnings": warnings,
        "messages": messages,
        "defaults": {
            "risk_free": (risk_free_default, risk_free_source),
            "beta": (beta_default, beta_source),
            "tax": (tax_default, tax_source),
            "growth": (growth_default, growth_source),
            "terminal_growth": (terminal_default, terminal_source),
            "roic": (roic_default, roic_source),
            "margin": (target_margin_default, margin_source),
            "current_margin": (
                current_margin_default,
                "TTM margin, otherwise most recent matched-period margin",
            ),
            "debt_weight": (debt_weight, debt_weight_source),
        },
    }


# ------------------------------------------------------------
# 4. Company loading and reset
# ------------------------------------------------------------
st.sidebar.header("1. Company")
ticker = st.sidebar.text_input(
    "Yahoo Finance ticker", value="AAPL",
    help="Examples: AAPL, D05.SI, 7203.T",
)

if "company_data" not in st.session_state:
    st.session_state.company_data = None

if st.sidebar.button("Load data"):
    if not ticker.strip():
        st.sidebar.error("Enter a ticker.")
    else:
        try:
            with st.spinner("Loading company data..."):
                loaded = load_company(ticker.strip().upper())
            st.session_state.company_data = loaded
            reset_defaults()
        except Exception as error:
            st.sidebar.error(f"Could not load company: {error}")

if st.session_state.company_data is None:
    st.info("Enter a ticker and click Load data.")
    st.stop()

data = st.session_state.company_data
defaults = data["defaults"]
currency = data["currency"]
financial_currency = data["financial_currency"]
statement_unit = financial_currency or "statement currency"

st.sidebar.button(
    "Reset to company defaults",
    on_click=reset_defaults,
    help="Resets all valuation inputs without downloading data again.",
)


# ------------------------------------------------------------
# 5. Snapshot and default diagnostics
# ------------------------------------------------------------
st.header(f"Company snapshot: {data['name']} ({data['ticker']})")

if is_financial_company(data["sector"], data["industry"]):
    st.warning(
        "This appears to be a bank, insurer, or REIT. A standard "
        "unlevered FCF DCF is not suitable without substantial adjustments."
    )

cols = st.columns(4)
cols[0].metric("Current price", format_money(data["price"], currency))
cols[1].metric(
    "Shares outstanding",
    f"{data['shares']:,.0f}" if data["shares"] is not None else "Not available",
)
cols[2].metric("Market capitalization", format_money(data["market_cap"], currency))
cols[3].metric("Sector", data["sector"])

cols = st.columns(4)
for col, label, field in zip(
    cols, ["Debt", "Cash", "Revenue", "Operating income"],
    ["debt", "cash", "revenue", "operating_income"],
):
    col.metric(label, format_money(data[field], financial_currency))

cols = st.columns(3)
cols[0].metric("Raw effective tax rate", pct(data["raw_tax"]))
cols[1].metric("Tax default", f"{defaults['tax'][0]:.2f}%")
cols[2].metric("Country", data["country"])

st.write(f"Operating income source: {data['operating_source']}")
st.write(f"Revenue source: {data['revenue_source']}")
st.write(f"Tax default source: {defaults['tax'][1]}")

cols = st.columns(4)
cols[0].metric(
    "Raw beta",
    f"{data['raw_beta']:.3f}" if data["raw_beta"] is not None else "Missing",
)
cols[1].metric("Blume-adjusted beta", f"{data['blume_beta']:.3f}")
cols[2].metric("Sector beta floor", f"{data['beta_floor']:.2f}")
cols[3].metric("Beta default after floor", f"{defaults['beta'][0]:.3f}")

cols = st.columns(3)
cols[0].metric("Raw three-year CAGR", pct(data["raw_cagr"]))
cols[1].metric("Raw historical ROIC", pct(data["raw_roic"]))
cols[2].metric("ROIC default", f"{defaults['roic'][0]:.2f}%")
st.caption(
    "Historical ROIC uses ending invested capital, not average invested "
    "capital. Historical ROIC is only a starting point for return on new capital."
)

with st.expander("Operating-margin default components", expanded=False):
    components = pd.DataFrame(data["margin_components"])
    if not components.empty:
        st.dataframe(
            components.style.format({"Margin": "{:.2%}"}, na_rep="Missing"),
            hide_index=True,
            use_container_width=True,
        )
    st.write(f"Margin default source: {defaults['margin'][1]}")

if data["operating_method"] == "EBIT":
    st.warning(
        "EBIT is used as an operating-income proxy. Review non-operating items."
    )
for message in data["warnings"]:
    st.warning(message)
if data["messages"]:
    with st.expander("Data retrieval messages"):
        for message in data["messages"]:
            st.write(message)


# ------------------------------------------------------------
# 6. Editable widgets and defaults audit
# ------------------------------------------------------------
audit = []


def record_default(label, default, current, source, percent=False):
    def display(value):
        if value is None:
            return "Missing: manual entry required"
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        if percent:
            return f"{value:,.4f}%"
        return f"{value:,.6g}" if isinstance(value, (float, int)) else str(value)

    audit.append({
        "Input": label,
        "Company default": display(default),
        "Current input": display(current),
        "Source / derivation": source,
    })


def number(label, default, source, key, minimum=None, maximum=None,
           step=0.1, percent=False):
    value = st.sidebar.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=float(default) if default is not None else None,
        step=float(step),
        key=f"dcf_{key}",
    )
    record_default(label, default, value, source, percent)
    return value


def company_number(label, field, minimum=None, step=0.1, maximum=None):
    value, source = defaults[field]
    return number(
        label, value, source, field, minimum, maximum, step, percent=True
    )


st.sidebar.header("2. DCF assumptions")
growth_percent = company_number(
    "Starting revenue growth: Year 1 (%)", "growth", minimum=-99.0, step=0.5
)
st.sidebar.caption(defaults["growth"][1])

if data["analyst_rows"]:
    st.sidebar.caption("Analyst revenue growth: reference only")
    for reference in data["analyst_rows"]:
        st.sidebar.write(
            f"{reference['Period']}: {reference['Revenue growth']:.2%}"
        )
    with st.expander("Analyst revenue-growth references"):
        st.dataframe(
            pd.DataFrame(data["analyst_rows"]).style.format(
                {"Revenue growth": "{:.2%}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
else:
    st.sidebar.caption("Analyst revenue-growth reference: unavailable.")

target_margin_percent = company_number(
    "Target operating margin (%)", "margin", step=0.5
)
st.sidebar.caption(defaults["margin"][1])

fade_margin = st.sidebar.checkbox(
    "Move operating margin gradually to target",
    value=False, key="dcf_fade_margin",
)
record_default(
    "Gradual margin transition", False, fade_margin,
    "Preserved model default: disabled",
)

# Always editable, even when the transition is currently disabled.
current_margin_percent = company_number(
    "Current operating margin (%)", "current_margin", step=0.5
)
st.sidebar.caption(
    "Used as Year 0 margin when transition is enabled. "
    "One fifth of the gap closes each year."
)

tax_percent = company_number(
    "Tax rate (%)", "tax", minimum=0.0, maximum=100.0, step=0.5
)
roic_percent = company_number(
    "Return on new invested capital (ROIC) (%)",
    "roic", minimum=0.01, step=0.5,
)
st.sidebar.caption(defaults["roic"][1])

risk_free_percent = company_number(
    "Risk-free rate (%)", "risk_free", minimum=0.0
)
st.sidebar.caption(
    "Editable government-yield placeholder based on financial-statement "
    "currency, not trading currency."
)
erp_percent = number(
    "Equity risk premium (%)", 5.0, "Illustrative manual assumption",
    "erp", minimum=0.0, percent=True,
)
beta = number(
    "Beta used in WACC", defaults["beta"][0], defaults["beta"][1],
    "beta", step=0.05,
)
st.sidebar.caption(
    "The sector floor sets the default only. You may override beta manually."
)
debt_cost_percent = number(
    "Cost of debt (%)", 5.0, "Illustrative pre-tax borrowing-cost assumption",
    "cost_debt", minimum=0.0, percent=True,
)
debt_weight_percent = company_number(
    "Debt weight in WACC (%)", "debt_weight",
    minimum=0.0, maximum=100.0, step=1.0,
)

override_wacc = st.sidebar.checkbox(
    "Override WACC manually", value=False, key="dcf_override"
)
record_default(
    "Manual WACC override", False, override_wacc,
    "Preserved model default: calculate WACC",
)
manual_wacc_percent = number(
    "Manual WACC (%)", 8.0, "Illustrative override value; used only if enabled",
    "manual_wacc", minimum=0.01, percent=True,
)

terminal_requested = company_number(
    "Terminal growth rate (%)", "terminal_growth", minimum=-99.0
)
terminal_used_percent = min(terminal_requested, risk_free_percent - 0.5)
if terminal_used_percent != terminal_requested:
    st.sidebar.warning(
        f"Terminal growth capped at risk-free minus 0.50 percentage points. "
        f"Actual rate used: {terminal_used_percent:.2f}%."
    )


# ------------------------------------------------------------
# 7. Base financial inputs and equity-bridge adjustments
# ------------------------------------------------------------
st.sidebar.header("3. Base financial data")
st.sidebar.caption("Enter full amounts, not millions.")


def base_number(label, field, source):
    value = data[field]
    if value is None:
        st.sidebar.warning(f"{label} is missing; enter it manually.")
    return number(
        label, value, source, f"base_{field}",
        minimum=0.0, step=1.0,
    )


starting_revenue = base_number("Starting revenue", "revenue", data["revenue_source"])
total_debt = base_number("Total debt", "debt", "Yahoo totalDebt")
total_cash = base_number("Cash", "cash", "Yahoo totalCash")
shares = base_number("Shares outstanding", "shares", "Yahoo sharesOutstanding")
current_price = base_number("Current share price", "price", "Yahoo currentPrice")

same_currency = (
    bool(financial_currency) and bool(currency)
    and financial_currency == currency
)
fx_default = 1.0 if same_currency else None
if not same_currency:
    st.sidebar.warning(
        "Statement and quote currencies differ or are unknown. "
        "Enter the conversion factor; do not assume parity."
    )
fx_factor = number(
    f"Quote units per 1 {statement_unit}",
    fx_default,
    "1.0 for identical known currencies; otherwise manual FX required",
    "fx", minimum=0.0, step=0.01,
)
st.sidebar.caption(
    "Use a compatible share basis. ADRs/depositary receipts may "
    "require a share-count adjustment."
)

st.sidebar.header("4. Net debt adjustments")
st.sidebar.caption(
    f"All amounts in {statement_unit}. Enter positive amounts. "
    "These are incremental adjustments, not automatically retrieved balances."
)
minority = number(
    "Minority interests", 0.0, "Manual adjustment: zero until reviewed",
    "minority", minimum=0.0, step=1.0,
)
pension = number(
    "Pension deficit", 0.0, "Manual adjustment: zero until reviewed",
    "pension", minimum=0.0, step=1.0,
)
leases = number(
    "Lease liabilities", 0.0,
    "Manual incremental lease adjustment: zero until reviewed",
    "leases", minimum=0.0, step=1.0,
)
investments = number(
    "Associates / investments", 0.0,
    "Manual non-operating investment value: zero until reviewed",
    "investments", minimum=0.0, step=1.0,
)
st.sidebar.warning(
    "Avoid double counting: debt may already include leases. Add lease "
    "liabilities only if not already included and consistent with your "
    "operating cash flows. Add investments only if their earnings are "
    "excluded from operating cash flows."
)

with st.sidebar.expander("Reverse DCF search bounds"):
    wacc_lower = number(
        "Minimum WACC to search (%)",
        max(0.01, defaults["terminal_growth"][0] + 0.01),
        "Default terminal growth + 0.01 percentage points, minimum 0.01%",
        "reverse_wacc_low", minimum=0.01, percent=True,
    )
    wacc_upper = number(
        "Maximum WACC to search (%)", 50.0,
        "Numerical search assumption", "reverse_wacc_high",
        minimum=0.01, percent=True,
    )
    growth_lower = number(
        "Minimum constant growth to search (%)", -50.0,
        "Numerical search assumption", "reverse_growth_low",
        minimum=-99.0, step=1.0, percent=True,
    )
    growth_upper = number(
        "Maximum constant growth to search (%)", 100.0,
        "Numerical search assumption", "reverse_growth_high",
        minimum=-99.0, step=1.0, percent=True,
    )

with st.sidebar.expander("Sensitivity settings"):
    wacc_step = number(
        "WACC step (percentage points)", 0.5,
        "Sensitivity-grid spacing", "sensitivity_wacc_step",
        minimum=0.01,
    )
    terminal_step = number(
        "Terminal growth step (percentage points)", 0.25,
        "Sensitivity-grid spacing", "sensitivity_terminal_step",
        minimum=0.01, step=0.05,
    )

with st.expander("Defaults used and their sources"):
    st.caption(
        "Company defaults are calculated when data loads. Manual changes "
        "do not silently rewrite the historical defaults. Reset restores "
        "this table's defaults. Load data again to refresh Yahoo data."
    )
    st.dataframe(pd.DataFrame(audit), hide_index=True, use_container_width=True)
    st.write(
        f"Actual terminal growth after the current risk-free cap: "
        f"{terminal_used_percent:.2f}%."
    )
    st.caption(
        "The risk-free table contains editable placeholders, not verified "
        "live yields. Country tax rates are simplified modeling assumptions."
    )

required = {
    "starting revenue growth": growth_percent,
    "target operating margin": target_margin_percent,
    "starting revenue": starting_revenue,
    "debt": total_debt,
    "cash": total_cash,
    "shares": shares,
    "current price": current_price,
    "currency conversion": fx_factor,
}
if fade_margin:
    required["current operating margin"] = current_margin_percent

missing = [name for name, value in required.items() if value is None]
if missing:
    st.error("Manual input required: " + ", ".join(missing))
    st.stop()

if min(starting_revenue, shares, current_price, fx_factor) <= 0:
    st.error("Revenue, shares, price, and FX conversion must be positive.")
    st.stop()


# ------------------------------------------------------------
# 8. WACC and equity bridge
# ------------------------------------------------------------
growth = growth_percent / 100
margin = target_margin_percent / 100
current_margin = current_margin_percent / 100 if fade_margin else margin
tax = tax_percent / 100
roic = roic_percent / 100
risk_free = risk_free_percent / 100
terminal_growth = terminal_used_percent / 100
cost_equity = risk_free + beta * erp_percent / 100
after_tax_debt_cost = debt_cost_percent / 100 * (1 - tax)
weight_debt = debt_weight_percent / 100
calculated_wacc = (
    (1 - weight_debt) * cost_equity + weight_debt * after_tax_debt_cost
)
wacc = manual_wacc_percent / 100 if override_wacc else calculated_wacc

# Positive deductions reduce equity value; investments increase it.
net_debt = total_debt - total_cash
bridge_deductions = net_debt + minority + pension + leases - investments

st.header("DCF assumptions and results")
st.subheader("Cost of capital")
cols = st.columns(4)
for col, label, value in zip(
    cols,
    ["Cost of equity", "After-tax cost of debt", "Calculated WACC", "WACC used"],
    [cost_equity, after_tax_debt_cost, calculated_wacc, wacc],
):
    col.metric(label, pct(value))

st.caption(
    f"WACC source: {'manual override' if override_wacc else 'CAPM calculation'}. "
    f"Terminal growth used: {terminal_used_percent:.2f}%."
)
if wacc <= terminal_growth or wacc <= -1:
    st.error("WACC must exceed terminal growth and be above -100%.")
    st.stop()


# ------------------------------------------------------------
# 9. Shared valuation engine
# ------------------------------------------------------------
def calculate_dcf(starting_growth, discount_rate, perpetual_growth,
                  constant_growth=False):
    """
    Reinvestment in Year t funds incremental NOPAT in Year t+1.
    All NOPAT improvements, including margin improvements, require capital.
    Explicit-period declines do not automatically release capital.
    """
    if (
        starting_growth <= -1 or perpetual_growth <= -1
        or discount_rate <= -1 or discount_rate <= perpetual_growth
        or roic <= 0
    ):
        return None

    years = np.arange(1, 6)
    growth_rates = (
        np.full(5, starting_growth) if constant_growth
        else np.linspace(starting_growth, perpetual_growth, 5)
    )
    margins = (
        np.linspace(current_margin, margin, 6)[1:]
        if fade_margin else np.full(5, margin)
    )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        revenues = starting_revenue * np.cumprod(1 + growth_rates)
        operating_income = revenues * margins
        taxes = operating_income * tax
        nopat = operating_income - taxes
        terminal_nopat = nopat[-1] * (1 + perpetual_growth)
        next_nopat = np.concatenate([nopat[1:], [terminal_nopat]])
        reinvestment = np.maximum(next_nopat - nopat, 0) / roic
        fcf = nopat - reinvestment
        discount_factors = 1 / (1 + discount_rate) ** years
        pv_fcf = fcf * discount_factors

        terminal_reinvestment_rate = perpetual_growth / roic
        terminal_reinvestment = terminal_nopat * terminal_reinvestment_rate
        terminal_fcf = terminal_nopat - terminal_reinvestment
        terminal_value = terminal_fcf / (discount_rate - perpetual_growth)
        pv_terminal = terminal_value / (1 + discount_rate) ** 5
        enterprise_value = float(pv_fcf.sum() + pv_terminal)
        equity_value = enterprise_value - bridge_deductions
        per_share = equity_value * fx_factor / shares

    check = np.concatenate([
        revenues, nopat, reinvestment, fcf, pv_fcf,
        [terminal_fcf, enterprise_value, equity_value, per_share],
    ])
    if not np.isfinite(check).all():
        return None

    ratios = np.divide(
        reinvestment, nopat,
        out=np.full(5, np.nan), where=nopat > 0,
    )
    forecast = pd.DataFrame({
        "Year": years,
        "Revenue growth": growth_rates,
        "Operating margin": margins,
        "Revenue": revenues,
        "Operating income": operating_income,
        "Taxes": taxes,
        "NOPAT": nopat,
        "Reinvestment": reinvestment,
        "Reinvestment / NOPAT": ratios,
        "Unlevered FCF": fcf,
        "Discount factor": discount_factors,
        "Present value of FCF": pv_fcf,
    })
    return {
        "forecast": forecast,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "per_share": float(per_share),
        "pv_terminal": float(pv_terminal),
        "terminal_value": float(terminal_value),
        "terminal_nopat": float(terminal_nopat),
        "terminal_fcf": float(terminal_fcf),
        "terminal_reinvestment": float(terminal_reinvestment),
        "terminal_reinvestment_rate": float(terminal_reinvestment_rate),
    }


def value_at(starting_growth, discount_rate, perpetual_growth=None,
             constant_growth=False):
    return calculate_dcf(
        starting_growth, discount_rate,
        terminal_growth if perpetual_growth is None else perpetual_growth,
        constant_growth,
    )


result = value_at(growth, wacc)
if result is None:
    st.error("Inputs did not produce a finite valuation.")
    st.stop()

forecast_df = result["forecast"]
enterprise_value = result["enterprise_value"]
equity_value = result["equity_value"]
value_per_share = result["per_share"]
upside = (value_per_share / current_price - 1) * 100
terminal_share = (
    result["pv_terminal"] / enterprise_value if enterprise_value > 0 else None
)
base_nopat = starting_revenue * current_margin * (1 - tax)
base_reinvestment = max(forecast_df["NOPAT"].iloc[0] - base_nopat, 0) / roic
starting_fcf = base_nopat - base_reinvestment


# ------------------------------------------------------------
# 10. Results and warnings
# ------------------------------------------------------------
st.subheader("Valuation result")
cols = st.columns(3)
cols[0].metric("Enterprise value", format_money(enterprise_value, financial_currency))
cols[1].metric("Equity value", format_money(equity_value, financial_currency))
cols[2].metric("Value per share", f"{value_per_share:,.2f} {currency}")

cols = st.columns(3)
cols[0].metric("Current price", f"{current_price:,.2f} {currency}")
cols[1].metric("Upside / downside", f"{upside:,.2f}%")
cols[2].metric(
    "Terminal value share of EV",
    pct(terminal_share) if terminal_share is not None else "Not meaningful",
)

st.subheader("Equity bridge")
bridge = pd.DataFrame({
    "Item": [
        "Enterprise value",
        "Less: total debt",
        "Add: cash",
        "Less: minority interests",
        "Less: pension deficit",
        "Less: incremental lease liabilities",
        "Add: associates / investments",
        "Equity value",
    ],
    f"Amount ({statement_unit})": [
        enterprise_value, -total_debt, total_cash,
        -minority, -pension, -leases, investments, equity_value,
    ],
})
st.dataframe(
    bridge.style.format({f"Amount ({statement_unit})": "{:,.2f}"}),
    hide_index=True, use_container_width=True,
)

st.subheader("Assumptions actually used")
cols = st.columns(4)
cols[0].metric("Target operating margin", f"{target_margin_percent:.2f}%")
cols[1].metric("Tax rate", f"{tax_percent:.2f}%")
cols[2].metric("Starting FCF: model Year 0", format_money(starting_fcf, financial_currency))
cols[3].metric(
    "Forecast Year 1 FCF",
    format_money(forecast_df["Unlevered FCF"].iloc[0], financial_currency),
)
cols = st.columns(3)
cols[0].metric("Return on new capital", f"{roic_percent:.2f}%")
cols[1].metric("Terminal reinvestment rate", pct(result["terminal_reinvestment_rate"]))
cols[2].metric("Terminal Year 6 FCF", format_money(result["terminal_fcf"], financial_currency))

st.write(
    f"Revenue growth fades from {growth_percent:.2f}% in Year 1 to "
    f"{terminal_used_percent:.2f}% in Year 5. "
    f"Margin transition: {'enabled' if fade_margin else 'disabled'}."
)
st.caption(
    "Starting FCF is model-derived, not historical reported FCF. "
    "Reinvestment funds next year's incremental NOPAT, including "
    "margin-driven improvements, at the selected ROIC."
)

st.subheader("Warnings")
valuation_warnings = []
if terminal_share is not None and terminal_share > 0.75:
    valuation_warnings.append(
        f"Terminal value is {terminal_share:.1%} of enterprise value. "
        "Most value depends on assumptions beyond Year 5."
    )
if abs(upside) > 50:
    valuation_warnings.append(
        "Result is extreme. Check data inputs and assumptions before trusting it."
    )
if enterprise_value <= 0:
    valuation_warnings.append("Enterprise value is not positive; terminal-value share is not meaningful.")
if margin <= 0:
    valuation_warnings.append("Terminal operating margin is not positive; perpetual growth may not be meaningful.")
if (forecast_df["NOPAT"] <= 0).any():
    valuation_warnings.append(
        "Forecast includes non-positive NOPAT. This is not a full "
        "turnaround model; tax-loss carryforwards are not modeled."
    )
if (forecast_df["Unlevered FCF"] < 0).any():
    valuation_warnings.append("Some forecast FCF is negative because reinvestment exceeds NOPAT.")
if result["terminal_reinvestment_rate"] >= 1:
    valuation_warnings.append("Terminal growth is at least as high as ROIC; terminal reinvestment consumes all or more of NOPAT.")
if terminal_growth < 0:
    valuation_warnings.append(
        "Negative terminal growth implies capital releases through g / ROIC. "
        "Explicit-period reinvestment is instead floored at zero."
    )
if fade_margin and margin != current_margin:
    valuation_warnings.append(
        "All incremental NOPAT is treated as capital-driven. Pure "
        "efficiency gains may require less capital than this model assumes."
    )
if valuation_warnings:
    for message in valuation_warnings:
        st.warning(message)
else:
    st.success("No major calculation warnings.")


# ------------------------------------------------------------
# 11. Cash-flow table and chart
# ------------------------------------------------------------
st.subheader("Projected cash flows")
money_columns = [
    "Revenue", "Operating income", "Taxes", "NOPAT",
    "Reinvestment", "Unlevered FCF", "Present value of FCF",
]
display_df = forecast_df.copy()
display_df[money_columns] /= 1_000_000
display_df.rename(
    columns={name: f"{name} (M)" for name in money_columns}, inplace=True
)
formats = {f"{name} (M)": "{:,.2f}" for name in money_columns}
formats.update({
    "Revenue growth": "{:.2%}",
    "Operating margin": "{:.2%}",
    "Reinvestment / NOPAT": "{:.2%}",
    "Discount factor": "{:.4f}",
})
st.dataframe(
    display_df.style.format(formats, na_rep="Not meaningful"),
    hide_index=True, use_container_width=True,
)
st.caption(f"Money values are in millions of {statement_unit}.")

chart = go.Figure()
for name, color in [
    ("Unlevered FCF", "#1f77b4"),
    ("Present value of FCF", "#ff7f0e"),
]:
    chart.add_trace(go.Bar(
        x=forecast_df["Year"].astype(str), y=forecast_df[name],
        name=name, marker_color=color,
    ))
chart.update_layout(
    title="Unlevered free cash flow and present value",
    xaxis_title="Forecast year", yaxis_title=f"Amount ({statement_unit})",
    barmode="group", template="plotly_white",
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(chart, use_container_width=True)


# ------------------------------------------------------------
# 12. Reverse DCF
# ------------------------------------------------------------
def solve_bisection(function, lower, upper, target):
    low_value, high_value = function(lower), function(upper)
    if low_value is None or high_value is None:
        return None

    low_error, high_error = low_value - target, high_value - target
    tolerance = max(abs(target) * 1e-8, 1e-8)
    if abs(low_error) <= tolerance:
        return lower
    if abs(high_error) <= tolerance:
        return upper
    if np.sign(low_error) == np.sign(high_error):
        return None

    for _ in range(200):
        middle = (lower + upper) / 2
        value = function(middle)
        if value is None:
            return None
        error = value - target
        if abs(error) <= tolerance:
            return middle
        if np.sign(error) == np.sign(low_error):
            lower, low_error = middle, error
        else:
            upper = middle
    return None


def find_roots(function, lower, upper, target):
    roots = []
    previous_x = None
    previous_value = None
    tolerance = max(abs(target) * 1e-8, 1e-8)

    for point in np.linspace(lower, upper, 401):
        point = float(point)
        value = function(point)
        if value is None or not np.isfinite(value):
            previous_x, previous_value = None, None
            continue
        error = value - target
        if abs(error) <= tolerance:
            roots.append(point)
        elif (
            previous_value is not None
            and np.sign(error) != np.sign(previous_value - target)
        ):
            root = solve_bisection(function, previous_x, point, target)
            if root is not None:
                roots.append(root)
        previous_x, previous_value = point, value

    unique = []
    for root in sorted(roots):
        if not unique or abs(root - unique[-1]) > 1e-6:
            unique.append(root)
    return unique


def reverse_price(candidate, mode):
    candidate_result = (
        value_at(growth, candidate) if mode == "WACC"
        else value_at(candidate, wacc, constant_growth=True)
    )
    return None if candidate_result is None else candidate_result["per_share"]


st.subheader("Reverse DCF")
st.write(
    "Solve separately for WACC and constant five-year revenue growth. "
    "Each scenario includes the same equity-bridge adjustments."
)
reverse_columns = st.columns(2)
for container, mode, lower, upper in [
    (reverse_columns[0], "WACC", wacc_lower, wacc_upper),
    (reverse_columns[1], "Constant growth", growth_lower, growth_upper),
]:
    with container:
        st.write(f"Solve for: {mode}")
        if lower >= upper:
            st.error("Lower search bound must be below upper bound.")
            continue
        if mode == "WACC" and lower / 100 <= terminal_growth:
            st.error("Minimum WACC search bound must exceed terminal growth.")
            continue
        roots = find_roots(
            lambda candidate: reverse_price(candidate, mode),
            lower / 100, upper / 100, current_price,
        )
        if not roots:
            st.warning("No price-matching solution found. Review or widen the sidebar bounds.")
        for root in roots:
            st.metric(f"Implied {mode.lower()}", pct(root))
            st.write(
                f"Matched value per share: "
                f"{reverse_price(root, mode):,.4f} {currency}"
            )
        if len(roots) > 1:
            st.warning("Multiple solutions found; ROIC-linked growth valuation can be non-monotonic.")

st.caption(
    "Implied WACC uses the main model's fading-growth path. Implied "
    "constant growth holds revenue growth fixed in Years 1–5, then "
    "switches to terminal growth in Year 6. All other inputs stay fixed. "
    "The scan may miss touching roots or very closely spaced solutions."
)


# ------------------------------------------------------------
# 13. Sensitivity heatmap
# ------------------------------------------------------------
st.subheader("Value-per-share sensitivity")
wacc_values = wacc + np.arange(-3, 4) * wacc_step / 100
terminal_values = terminal_growth + np.arange(-3, 4) * terminal_step / 100
sensitivity = np.full((7, 7), np.nan)

for row, candidate_wacc in enumerate(wacc_values):
    for col, candidate_growth in enumerate(terminal_values):
        if (
            candidate_wacc <= 0 or candidate_wacc <= candidate_growth
            or candidate_growth > risk_free - 0.005 + 1e-12
            or candidate_growth <= -1
        ):
            continue
        cell = value_at(growth, float(candidate_wacc), float(candidate_growth))
        if cell is not None:
            sensitivity[row, col] = cell["per_share"]

wacc_labels = [f"{value * 100:.2f}%" for value in wacc_values]
growth_labels = [f"{value * 100:.2f}%" for value in terminal_values]
heatmap = go.Figure(go.Heatmap(
    z=sensitivity, x=growth_labels, y=wacc_labels,
    colorscale="RdYlGn", zmid=current_price,
    colorbar=dict(title=f"Value/share {currency}"),
    hoverongaps=False,
    hovertemplate=(
        "WACC: %{y}<br>Terminal growth: %{x}<br>"
        "Value/share: %{z:,.2f}<extra></extra>"
    ),
))
for row, row_label in enumerate(wacc_labels):
    for col, col_label in enumerate(growth_labels):
        value = sensitivity[row, col]
        text = (
            f"{value:,.2f}" + (" ★" if value > current_price else "")
            if np.isfinite(value) else "Invalid"
        )
        heatmap.add_annotation(
            x=col_label, y=row_label, text=text, showarrow=False,
            font=dict(color="black", size=12),
            bgcolor="rgba(255,255,255,0.70)",
        )
heatmap.update_layout(
    title="Value per share: WACC versus terminal growth",
    xaxis_title="Terminal growth", yaxis_title="WACC",
    yaxis=dict(autorange="reversed"),
    template="plotly_white", height=530,
)
st.plotly_chart(heatmap, use_container_width=True)
st.caption(
    f"★ marks cells above the current price of {current_price:,.2f} {currency}. "
    "Every cell recalculates the growth fade, reinvestment, terminal FCF, "
    "and equity bridge. Cells violating WACC > growth or the "
    "risk-free-minus-0.50-percentage-point cap are invalid."
)


# ------------------------------------------------------------
# 14. Explanation and limitations
# ------------------------------------------------------------
st.subheader("How this DCF works")
st.write(
    """
1. Revenue growth fades from the starting rate to terminal growth in Year 5.
2. Margin either stays at target or moves from current margin to target by Year 5.
3. Operating income less modeled taxes produces NOPAT.
4. Reinvestment funds next year's incremental NOPAT at the assumed ROIC.
5. Explicit-period reinvestment is floored at zero.
6. Terminal reinvestment rate is terminal growth divided by ROIC.
7. Forecast FCF and terminal value are discounted at WACC.
8. Debt, minority interests, pension deficit, and incremental leases reduce equity value.
9. Cash and non-operating associates/investments increase equity value.
10. Equity value is converted to quote currency and divided by shares.
    """
)
st.caption(
    "Limitations: five forecast years; one ROIC for forecast and perpetuity; "
    "ending rather than average invested capital for historical ROIC; "
    "immediate tax effects; no separate capex, depreciation, working-capital, "
    "tax-loss, or dilution schedules. Margin-driven NOPAT changes are "
    "treated as requiring capital. All cash is treated as available to "
    "equity holders. Country is Yahoo's company country, not necessarily "
    "the tax jurisdiction of all earnings. Government yields and statutory "
)
