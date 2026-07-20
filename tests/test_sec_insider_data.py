from datetime import date

import pandas as pd

from tradingagents.dataflows import sec_insider_data


FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>Example Executive</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>true</isDirector><isOfficer>true</isOfficer>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-07-15</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>250</value></transactionShares>
        <transactionPricePerShare><value>100</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-07-16</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode><aff10b5One>1</aff10b5One></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>20000</value></transactionShares>
        <transactionPricePerShare><value>100</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-07-16</value></transactionDate>
      <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>999</value></transactionShares>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <derivativeTable>
    <derivativeTransaction>
      <transactionDate><value>2026-07-16</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
    </derivativeTransaction>
  </derivativeTable>
</ownershipDocument>"""


def test_form4_parser_keeps_only_open_market_non_derivative_trades():
    records = sec_insider_data.parse_form4_xml(
        FORM4_XML,
        ticker="AAPL",
        filing_date="2026-07-17",
        accession_number="0001-26-000001",
        filing_url="https://www.sec.gov/example.xml",
        as_of_date=date(2026, 7, 19),
    )

    assert [record["transaction_code"] for record in records] == ["P", "S"]
    assert records[0]["owner"] == "Example Executive"
    assert "Chief Executive Officer" in records[0]["role"]
    assert records[0]["value"] == 25_000
    assert records[0]["signal_score"] > 0
    assert records[1]["planned_10b5_1"] is True
    assert records[1]["signal_score"] == -1


def test_form4_parser_excludes_transactions_after_analysis_date():
    records = sec_insider_data.parse_form4_xml(
        FORM4_XML,
        ticker="AAPL",
        filing_date="2026-07-17",
        accession_number="0001-26-000001",
        filing_url="https://www.sec.gov/example.xml",
        as_of_date=date(2026, 7, 15),
    )

    assert [record["transaction_date"] for record in records] == ["2026-07-15"]


def test_recent_filings_enforces_publication_cutoff_and_limit():
    submissions = {
        "filings": {
            "recent": {
                "form": ["4", "4/A", "10-Q", "4"],
                "filingDate": ["2026-07-18", "2026-07-10", "2026-07-09", "2026-05-01"],
                "accessionNumber": ["future", "eligible", "wrong-form", "old"],
                "primaryDocument": ["future.xml", "eligible.xml", "quarterly.htm", "old.xml"],
            }
        }
    }

    filings = sec_insider_data._recent_form4_filings(
        submissions,
        as_of_date=date(2026, 7, 17),
        lookback_days=30,
        max_filings=1,
    )

    assert filings == []


def test_summary_rewards_purchase_and_only_weakly_penalizes_planned_sale():
    records = sec_insider_data.parse_form4_xml(
        FORM4_XML,
        ticker="AAPL",
        filing_date="2026-07-17",
        accession_number="0001-26-000001",
        filing_url="https://www.sec.gov/example.xml",
        as_of_date=date(2026, 7, 19),
    )

    summary = sec_insider_data.summarize_sec_insider_activity(pd.DataFrame(records))

    assert summary["purchase_count"] == 1
    assert summary["sale_count"] == 1
    assert summary["purchase_value"] == 25_000
    assert summary["sale_value"] == 2_000_000
    assert summary["signal_score"] > 0


def test_missing_sec_identity_fails_neutral_without_network(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sec_insider_data,
        "get_config",
        lambda: {
            "data_cache_dir": str(tmp_path),
            "sec_user_agent": "",
            "insider_lookback_days": 30,
            "insider_cache_hours": 12,
            "insider_max_filings": 20,
        },
    )
    monkeypatch.delenv("TRADINGAGENTS_SEC_USER_AGENT", raising=False)
    monkeypatch.setattr(
        sec_insider_data.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network should not be called without an SEC identity")
        ),
    )

    activity = sec_insider_data.get_sec_insider_activity(
        "AAPL", as_of_date="2026-07-19"
    )

    assert activity.empty
    assert activity.attrs["data_status"] == "unavailable"
