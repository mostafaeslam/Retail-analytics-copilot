"""DSPy-compatible function signatures and modules.

This file defines DSPy Signatures and Modules for Router, NL->SQL, and Synthesizer.
We'll optimize at least one module using DSPy optimizers.
"""

import re
from typing import Dict, Any

import dspy


# Router: classifies into 'rag' | 'sql' | 'hybrid'
class RouterSignature(dspy.Signature):
    """Classify the question type to determine which pipeline to use.

    - 'rag': Pure RAG questions about policies, definitions, or information from documents
    - 'sql': Pure SQL questions that only need database queries
    - 'hybrid': Questions that need both document context (dates, definitions) and SQL queries
    """

    question: str = dspy.InputField(desc="The user's question")
    route: str = dspy.OutputField(desc="One of: 'rag', 'sql', or 'hybrid'")


# NL->SQL: generate SQL queries from planner hints
class NLToSQLSignature(dspy.Signature):
    """Generate a SQLite query based on the planner's intent and extracted constraints.

    The SQL should:
    - Use correct table names: Orders, "Order Details", Products, Categories, Customers
    - Join tables properly using OrderID, ProductID, CategoryID
    - Apply date filters using DATE(OrderDate)
    - Calculate metrics correctly (revenue, AOV, gross margin)
    - Use proper SQLite syntax
    """

    planner: str = dspy.InputField(
        desc="JSON string with intent, dates, category, metric, filters"
    )
    schema_hint: str = dspy.InputField(desc="Brief schema information for context")
    sql: str = dspy.OutputField(desc="Valid SQLite query ending with semicolon")


# Synthesizer: format outputs according to format_hint
class SynthesizerSignature(dspy.Signature):
    """Synthesize the final answer from SQL results and format according to format_hint.

    Extract the answer from the SQL results and format it exactly as specified:
    - int: return a single integer
    - float: return a float rounded to 2 decimals
    - object: return a dict with specified keys
    - list: return a list of objects

    Always include citations from both database tables and document chunks used.
    """

    format_hint: str = dspy.InputField(
        desc="The expected output format (int, float, object, list)"
    )
    sql_rows: str = dspy.InputField(desc="JSON string of SQL query results")
    doc_chunks: str = dspy.InputField(desc="List of document chunk IDs used")
    final_answer: str = dspy.OutputField(
        desc="The formatted final answer as JSON string"
    )
    explanation: str = dspy.OutputField(
        desc="Brief explanation (1-2 sentences) of how the answer was derived"
    )


# Simple fallback router (rule-based) for when DSPy is not configured
def router_classify_simple(question: str) -> str:
    """Simple rule-based router fallback."""
    q = question.lower()
    # Check for SQL/hybrid patterns first (these are more specific)
    if (
        "top" in q
        or "aov" in q
        or "average order value" in q
        or "revenue" in q
        or "gross margin" in q
        or "margin" in q
        or "quantity" in q
        or "sold" in q
        or "customer" in q
        or "product" in q
        or "category" in q
    ):
        # Check if it's a pure RAG question about return policy
        if ("return window" in q or "return days" in q) and (
            "policy" in q or "unopened" in q
        ):
            return "rag"
        # Otherwise it's hybrid (needs SQL + docs for dates/definitions)
        if "top" in q or "aov" in q or "revenue" in q or "margin" in q:
            return "hybrid"
        return "hybrid"
    # Pure RAG questions about policies
    if ("return window" in q or "return days" in q) and "policy" in q:
        return "rag"
    # Default to hybrid for questions that might need SQL
    return "hybrid"


# Router module (can be optimized with DSPy)
class Router(dspy.Module):
    def __init__(self):
        super().__init__()
        # Check if DSPy LM is configured
        try:
            # Try to access settings to see if LM is configured
            if (
                hasattr(dspy, "settings")
                and hasattr(dspy.settings, "lm")
                and dspy.settings.lm is not None
            ):
                self.classify = dspy.Predict(RouterSignature)
                self._dspy_configured = True
            else:
                self._dspy_configured = False
        except (AttributeError, RuntimeError, ValueError):
            # DSPy not configured, use fallback
            self._dspy_configured = False

    def forward(self, question: str) -> str:
        if not self._dspy_configured:
            return router_classify_simple(question)
        try:
            result = self.classify(question=question)
            route = result.route.lower().strip()
            if route in ["rag", "sql", "hybrid"]:
                return route
        except Exception:
            pass
        return router_classify_simple(question)


# NL->SQL module (can be optimized with DSPy)
class NLToSQL(dspy.Module):
    def __init__(self):
        super().__init__()
        # Check if DSPy LM is configured
        try:
            if (
                hasattr(dspy, "settings")
                and hasattr(dspy.settings, "lm")
                and dspy.settings.lm is not None
            ):
                self.generate = dspy.ChainOfThought(NLToSQLSignature)
                self._dspy_configured = True
            else:
                self._dspy_configured = False
        except (AttributeError, RuntimeError, ValueError):
            # DSPy not configured, use fallback
            self._dspy_configured = False

    def forward(self, planner: Dict[str, Any], schema_hint: str = "") -> str:
        if not self._dspy_configured:
            return self._fallback_sql(planner)
        try:
            planner_str = str(planner) if isinstance(planner, dict) else planner
            result = self.generate(planner=planner_str, schema_hint=schema_hint)
            sql = result.sql.strip()
            # Ensure it ends with semicolon
            if not sql.endswith(";"):
                sql += ";"
            return sql
        except Exception as e:
            # Fallback to rule-based SQL generation
            return self._fallback_sql(planner)

    def _fallback_sql(self, planner: Dict[str, Any]) -> str:
        """Fallback rule-based SQL generation."""
        intent = planner.get("intent", "")
        dates = planner.get("dates", [])
        category = planner.get("category")
        metric = planner.get("metric", "quantity")
        limit = planner.get("limit", 1)

        if intent == "top_category_qty" and dates:
            start, end = dates[0], dates[1]
            return (
                "SELECT c.CategoryName AS category, SUM(od.Quantity) AS quantity "
                'FROM "Order Details" od '
                "JOIN Orders o ON od.OrderID = o.OrderID "
                "JOIN Products p ON od.ProductID = p.ProductID "
                "JOIN Categories c ON p.CategoryID = c.CategoryID "
                f"WHERE o.OrderDate BETWEEN '{start}' AND '{end}' "
                "GROUP BY c.CategoryName ORDER BY quantity DESC LIMIT 1;"
            )
        elif intent == "aov" and dates:
            start, end = dates[0], dates[1]
            return (
                "SELECT CAST(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS REAL) / COUNT(DISTINCT o.OrderID) AS aov "
                'FROM "Order Details" od '
                "JOIN Orders o ON od.OrderID = o.OrderID "
                f"WHERE o.OrderDate BETWEEN '{start}' AND '{end}';"
            )
        elif intent == "gross_margin_customer" and dates:
            start, end = dates[0], dates[1]
            return (
                "SELECT c.CompanyName AS customer, "
                "SUM((od.UnitPrice - (od.UnitPrice * 0.7)) * od.Quantity * (1 - od.Discount)) AS margin "
                'FROM "Order Details" od '
                "JOIN Orders o ON od.OrderID = o.OrderID "
                "JOIN Customers c ON o.CustomerID = c.CustomerID "
                f"WHERE o.OrderDate BETWEEN '{start}' AND '{end}' "
                "GROUP BY c.CustomerID, c.CompanyName ORDER BY margin DESC LIMIT 1;"
            )
        elif intent == "gross_margin" and dates:
            start, end = dates[0], dates[1]
            return (
                "SELECT SUM((od.UnitPrice - (od.UnitPrice * 0.7)) * od.Quantity * (1 - od.Discount)) AS margin "
                'FROM "Order Details" od '
                "JOIN Orders o ON od.OrderID = o.OrderID "
                f"WHERE o.OrderDate BETWEEN '{start}' AND '{end}';"
            )
        elif intent == "top_products_revenue":
            return (
                "SELECT p.ProductName AS product, "
                "SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS revenue "
                'FROM "Order Details" od '
                "JOIN Products p ON od.ProductID = p.ProductID "
                "GROUP BY p.ProductID, p.ProductName "
                "ORDER BY revenue DESC LIMIT 3;"
            )
        elif intent == "category_revenue" and dates and category:
            start, end = dates[0], dates[1]
            return (
                "SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS revenue "
                'FROM "Order Details" od '
                "JOIN Orders o ON od.OrderID = o.OrderID "
                "JOIN Products p ON od.ProductID = p.ProductID "
                "JOIN Categories c ON p.CategoryID = c.CategoryID "
                f"WHERE c.CategoryName = '{category}' "
                f"AND o.OrderDate BETWEEN '{start}' AND '{end}';"
            )
        return ""


# Synthesizer module (can be optimized with DSPy)
class Synthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        # Check if DSPy LM is configured
        try:
            if (
                hasattr(dspy, "settings")
                and hasattr(dspy.settings, "lm")
                and dspy.settings.lm is not None
            ):
                self.synthesize = dspy.ChainOfThought(SynthesizerSignature)
                self._dspy_configured = True
            else:
                self._dspy_configured = False
        except (AttributeError, RuntimeError, ValueError):
            # DSPy not configured, use fallback
            self._dspy_configured = False

    def forward(
        self, format_hint: str, rows: list, doc_citations: list, table_citations: list
    ) -> Dict[str, Any]:
        if not self._dspy_configured:
            return self._fallback_synthesize(
                format_hint, rows, doc_citations, table_citations
            )
        try:
            rows_str = str(rows) if rows else "[]"
            doc_chunks_str = str(doc_citations) if doc_citations else "[]"
            result = self.synthesize(
                format_hint=format_hint, sql_rows=rows_str, doc_chunks=doc_chunks_str
            )

            # Parse the final_answer from JSON string
            try:
                import json

                final_answer = json.loads(result.final_answer)
            except:
                final_answer = self._parse_fallback(format_hint, rows)

            return {
                "final_answer": final_answer,
                "explanation": result.explanation,
                "citations": table_citations + doc_citations,
            }
        except Exception:
            # Fallback to rule-based synthesis
            return self._fallback_synthesize(
                format_hint, rows, doc_citations, table_citations
            )

    def _parse_fallback(self, format_hint: str, rows: list) -> Any:
        """Parse format_hint and extract answer from rows."""
        if not rows:
            return None

        if format_hint == "int":
            val = rows[0][list(rows[0].keys())[0]]
            if val is None:
                return None
            return int(float(val))  # Handle numeric strings
        elif format_hint == "float" or format_hint == "float_2":
            val = rows[0][list(rows[0].keys())[0]]
            if val is None:
                return None
            return round(float(val), 2)
        elif format_hint.startswith("{") and format_hint.endswith("}"):
            # Parse object format like "{category:str, quantity:int}"
            # Extract keys from format_hint
            inner = format_hint[1:-1]  # Remove {}
            parts = [k.strip() for k in inner.split(",")]
            keys = []
            type_hints = {}
            for part in parts:
                if ":" in part:
                    key, type_hint = part.split(":")
                    key = key.strip()
                    keys.append(key)
                    type_hints[key] = type_hint.strip()

            # Map SQL result columns to format hint keys
            result = {}
            row = rows[0]
            for key in keys:
                key_lower = key.lower()
                # Find matching column (case-insensitive)
                for col_name, col_value in row.items():
                    if key_lower == col_name.lower():
                        # Convert based on type hint
                        type_hint = type_hints.get(key, "str")
                        if "int" in type_hint:
                            result[key] = int(float(col_value))
                        elif "float" in type_hint:
                            result[key] = round(float(col_value), 2)
                        else:
                            result[key] = str(col_value)
                        break

            return result if result else rows[0]
        elif format_hint.startswith("list[") and format_hint.endswith("]"):
            # Parse list format like "list[{product:str, revenue:float}]"
            inner = format_hint[5:-1]  # Remove "list[" and "]"
            if inner.startswith("{") and inner.endswith("}"):
                # Extract keys and type hints from object format
                obj_inner = inner[1:-1]
                parts = [k.strip() for k in obj_inner.split(",")]
                keys = []
                type_hints = {}
                for part in parts:
                    if ":" in part:
                        key, type_hint = part.split(":")
                        key = key.strip()
                        keys.append(key)
                        type_hints[key] = type_hint.strip()

                result_list = []
                for row in rows:
                    result_obj = {}
                    for key in keys:
                        key_lower = key.lower()
                        for col_name, col_value in row.items():
                            if key_lower == col_name.lower():
                                type_hint = type_hints.get(key, "str")
                                if "int" in type_hint:
                                    result_obj[key] = int(float(col_value))
                                elif "float" in type_hint:
                                    result_obj[key] = round(float(col_value), 2)
                                else:
                                    result_obj[key] = str(col_value)
                                break
                    result_list.append(result_obj)
                return result_list
            return rows
        return rows

    def _fallback_synthesize(
        self, format_hint: str, rows: list, doc_citations: list, table_citations: list
    ) -> Dict[str, Any]:
        """Fallback rule-based synthesis."""
        final_answer = self._parse_fallback(format_hint, rows)
        explanation = (
            f"Answer extracted from SQL results matching format {format_hint}."
        )

        return {
            "final_answer": final_answer,
            "explanation": explanation,
            "citations": table_citations + doc_citations,
        }


# Wrapper functions for backward compatibility
def router_classify(question: str) -> str:
    """Router wrapper - uses DSPy if configured, else falls back."""
    try:
        router = Router()
        return router(question)
    except Exception:
        return router_classify_simple(question)


def nl_to_sql(planner: Dict[str, Any]) -> str:
    """NL->SQL wrapper - uses DSPy if configured, else falls back."""
    try:
        nl_to_sql_module = NLToSQL()
        return nl_to_sql_module(planner)
    except Exception:
        # Fallback
        nl_to_sql_module = NLToSQL()
        return nl_to_sql_module._fallback_sql(planner)


def synthesize(
    final_shape: Dict[str, Any], rows: Any, doc_citations: list, table_citations: list
) -> Dict[str, Any]:
    """Synthesizer wrapper - uses DSPy if configured, else falls back."""
    format_hint = final_shape.get("format_hint", "")
    try:
        synth_module = Synthesizer()
        return synth_module(format_hint, rows, doc_citations, table_citations)
    except Exception:
        # Fallback
        synth_module = Synthesizer()
        return synth_module._fallback_synthesize(
            format_hint, rows, doc_citations, table_citations
        )
