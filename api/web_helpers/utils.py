from dash import html
import dash_bootstrap_components as dbc

def _create_timeline_item(date, title, company, is_last=False) -> html.Div:
    return html.Div([
        html.Div([
            html.Div(className="timeline-point"),
            html.Div(className="timeline-line" if not is_last else "")
        ], className="timeline-middle"),
        html.Div([
            html.Div([
                html.Span(date, className="timeline-date me-3"),
                html.H5(title, className="d-inline mb-1"),
            ], className="d-flex align-items-center"),
            html.P(company, className="mb-0 text-muted")
        ], className="timeline-content")
    ], className="timeline-item d-flex")


def create_professional_timeline() -> html.Div:
    return html.Div([
        html.H2("Timeline", className="text-primary mb-4"),
        html.Div([
            _create_timeline_item("05/24", "Graduated with electrical engineering degree", "McGill University"),
            _create_timeline_item("01/23 - 04/24", "Data scientist intern: quantitative equities and derivatives strategies", "PSP Investments"),
            _create_timeline_item("05/22 - 08/22", "Data scientist intern: credit risk modelling", "National Bank of Canada"),
            _create_timeline_item("09/21 - 08/22", "Research assistant: unsupervised learning for drug discovery", "McGill University Health Center"),
            _create_timeline_item("09/19", "Start of electrical engineering degree", "McGill University", is_last=True)
        ], className="timeline")
    ], className="mb-5")

def create_experience_card(position: str, company: str, date: str, responsibilities: list[str]) -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader([
            html.H5(position, className="card-title mb-0"),
            html.H6(company, className="card-subtitle mt-1"),
        ]),
        dbc.CardBody([
            html.P(date, className="card-text text-muted mb-2"),
            html.Ul([html.Li(resp) for resp in responsibilities], className="mb-0 small")
        ])
    ], className="mb-3")
