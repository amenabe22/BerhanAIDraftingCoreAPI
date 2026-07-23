"""Unit tests for app/services/legal/instrument_router.py."""

import pytest

from app.services.legal.instrument_router import RouteDecision, route


class TestCommercialRouting:
    def test_company_query_routes_to_commercial(self):
        d = route("Can a company director bind the company without a board resolution?")
        assert "1243" in d.query_suffix or "Commercial" in d.query_suffix

    def test_shareholder_query_routes_to_commercial(self):
        d = route("What are shareholder rights in a private limited company?")
        assert "1243" in d.query_suffix or "Commercial" in d.query_suffix

    def test_commercial_forbidden_primary_excludes_civil_code(self):
        d = route("director board resolution company Ethiopia")
        assert any("civil" in f for f in d.forbidden_primary)

    def test_amharic_company_routes_to_commercial(self):
        d = route("ኩባንያ ዳይሬክተር ስልጣን")
        assert "1243" in d.query_suffix or "Commercial" in d.query_suffix


class TestLabourRouting:
    def test_employment_termination(self):
        d = route("Is it legal to terminate an employee without notice in Ethiopia?")
        assert "1156" in d.query_suffix or "Labour" in d.query_suffix

    def test_wages_query(self):
        d = route("What are the rules on minimum wages under Ethiopian law?")
        assert "Labour" in d.query_suffix or "1156" in d.query_suffix

    def test_amharic_worker_routes_labour(self):
        d = route("ሠራተኛ ስንብት ሕግ")
        assert "1156" in d.query_suffix or "Labour" in d.query_suffix


class TestTaxRouting:
    def test_income_tax_query(self):
        d = route("What income is exempt from income tax in Ethiopia?")
        assert "Tax" in d.query_suffix or "tax" in d.query_suffix.lower()

    def test_vat_query(self):
        d = route("How does VAT apply to small businesses?")
        assert "VAT" in d.query_suffix or "tax" in d.query_suffix.lower()

    def test_customs_query(self):
        d = route("What are the customs duties on imported electronics?")
        assert "customs" in d.query_suffix.lower()


class TestInheritanceRouting:
    def test_inheritance_query(self):
        d = route("How is inheritance distributed under Ethiopian law?")
        assert "1677" in d.query_suffix or "Civil Code" in d.query_suffix

    def test_limitation_period_query(self):
        d = route("What is the statute of limitations for a contract claim?")
        assert "1677" in d.query_suffix or "Civil Code" in d.query_suffix

    def test_amharic_inheritance(self):
        d = route("ውርስ የሚያወጣ ሕግ")
        assert "Civil Code" in d.query_suffix or "1677" in d.query_suffix


class TestCassationRouting:
    def test_cassation_english(self):
        d = route("Can I get the cassation ruling on this?")
        assert d.expect_kb_gap is True

    def test_cassation_amharic(self):
        d = route("ሰ/መ/ቁ. 33945 ውሳኔ")
        assert d.expect_kb_gap is True

    def test_seber_amharic(self):
        d = route("ሰበር ሰሚ ችሎት ውሳኔ")
        assert d.expect_kb_gap is True


class TestFISRouting:
    def test_fis_account_freeze(self):
        d = route("FIS አካውንት ለስንት ጊዜ ማገድ ይችላል?")
        assert d.expect_kb_gap is True

    def test_money_laundering(self):
        d = route("What are the penalties for money laundering in Ethiopia?")
        assert d.expect_kb_gap is True


class TestNeutralRouting:
    def test_neutral_query_returns_empty_suffix(self):
        d = route("Hello, how are you?")
        assert d.query_suffix == ""
        assert d.expect_kb_gap is False
        assert d.forbidden_primary == []

    def test_return_type(self):
        d = route("any query")
        assert isinstance(d, RouteDecision)
        assert isinstance(d.preferred_document_ids, list)
        assert isinstance(d.forbidden_primary, list)
