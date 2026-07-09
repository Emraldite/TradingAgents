from tradingagents.dataflows import congressional_data


def test_capitol_card_parser_handles_current_card_layout():
    html = """
    <html>
      <body>
        <h2>Kevin Hern</h2>
        <div>Republican House OK</div>
        <h3>Coterra Energy Inc</h3>
        <div>CTRA:US</div>
        <div>13:02</div>
        <div>Today</div>
        <div>8 May</div>
        <div>2026</div>
        <div>days</div>
        <div>25</div>
        <div>Joint</div>
        <div>exchange</div>
        <div>15K–50K</div>
        <div>N/A</div>
      </body>
    </html>
    """

    records = congressional_data._parse_capitol_trades_html(html)

    assert len(records) == 1
    assert records[0]["ticker"] == "CTRA"
    assert records[0]["politician"] == "Kevin Hern"
    assert records[0]["trade_type"] == "exchange"
    assert records[0]["disclosure_date"] == "2026-05-08"
    assert records[0]["amount"] == 50_000


def test_quiver_parser_accepts_generic_table_without_old_id():
    html = """
    <html>
      <body>
        <table>
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Type</th>
              <th>Politician</th>
              <th>Filed</th>
              <th>Traded</th>
              <th>Amount</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>NVDA</td>
              <td>buy</td>
              <td>Sheldon Whitehouse</td>
              <td>2026-05-09</td>
              <td>2026-05-08</td>
              <td>100K</td>
            </tr>
          </tbody>
        </table>
      </body>
    </html>
    """

    records = congressional_data._parse_quiver_html(html)

    assert len(records) == 1
    assert records[0]["ticker"] == "NVDA"
    assert records[0]["trade_type"] == "buy"
    assert records[0]["politician"] == "Sheldon Whitehouse"
    assert records[0]["disclosure_date"] == "2026-05-08"
    assert records[0]["amount"] == 100_000
