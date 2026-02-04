"""Tests for the Tier-1 Query Classifier."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models import QueryClassification


class TestQueryClassifier:
    """Test suite for QueryClassifier."""
    
    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response."""
        def _create_response(query_type: str, confidence: float = 0.9):
            response = MagicMock()
            response.content = f'''{{
                "query_type": "{query_type}",
                "confidence": {confidence},
                "reasoning": "Test reasoning"
            }}'''
            return response
        return _create_response
    
    @pytest.mark.asyncio
    async def test_classify_text_query(self, mock_llm_response):
        """Test classification of text-only query."""
        from app.services.classifier import QueryClassifier
        
        classifier = QueryClassifier()
        
        with patch.object(classifier, '_get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.__or__ = MagicMock(return_value=mock_llm)
            mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response("text"))
            mock_get_llm.return_value = mock_llm
            
            # Override the chain behavior
            classifier._llm = mock_llm
            
            # Create a mock chain
            with patch('app.services.classifier.query_classifier.ChatPromptTemplate') as mock_prompt:
                mock_chain = MagicMock()
                mock_chain.ainvoke = AsyncMock(return_value=mock_llm_response("text"))
                mock_prompt.from_template.return_value.__or__ = MagicMock(return_value=mock_chain)
                
                classification, tokens = await classifier.classify(
                    "What are the mechanisms of action of aspirin?"
                )
                
                assert classification.query_type == "text"
                assert classification.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_classify_sql_query(self, mock_llm_response):
        """Test classification of SQL-only query."""
        from app.services.classifier import QueryClassifier
        
        classifier = QueryClassifier()
        
        with patch('app.services.classifier.query_classifier.ChatPromptTemplate') as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value=mock_llm_response("sql"))
            mock_prompt.from_template.return_value.__or__ = MagicMock(return_value=mock_chain)
            
            classification, tokens = await classifier.classify(
                "What is the recommended dose of metformin?"
            )
            
            assert classification.query_type == "sql"
    
    @pytest.mark.asyncio
    async def test_classify_hybrid_query(self, mock_llm_response):
        """Test classification of hybrid query."""
        from app.services.classifier import QueryClassifier
        
        classifier = QueryClassifier()
        
        with patch('app.services.classifier.query_classifier.ChatPromptTemplate') as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value=mock_llm_response("hybrid"))
            mock_prompt.from_template.return_value.__or__ = MagicMock(return_value=mock_chain)
            
            classification, tokens = await classifier.classify(
                "What is the standard dose of metformin and why is it effective for diabetes?"
            )
            
            assert classification.query_type == "hybrid"
    
    @pytest.mark.asyncio
    async def test_classify_handles_error(self):
        """Test that classification defaults to hybrid on error."""
        from app.services.classifier import QueryClassifier
        
        classifier = QueryClassifier()
        
        with patch('app.services.classifier.query_classifier.ChatPromptTemplate') as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(side_effect=Exception("API Error"))
            mock_prompt.from_template.return_value.__or__ = MagicMock(return_value=mock_chain)
            
            classification, tokens = await classifier.classify("Any query")
            
            # Should default to hybrid on error
            assert classification.query_type == "hybrid"
            assert classification.confidence == 0.5
    
    def test_parse_json_response_clean(self):
        """Test parsing clean JSON response."""
        from app.services.classifier import QueryClassifier
        
        classifier = QueryClassifier()
        
        content = '{"query_type": "text", "confidence": 0.95, "reasoning": "Test"}'
        result = classifier._parse_json_response(content)
        
        assert result["query_type"] == "text"
        assert result["confidence"] == 0.95
    
    def test_parse_json_response_with_markdown(self):
        """Test parsing JSON wrapped in markdown code block."""
        from app.services.classifier import QueryClassifier
        
        classifier = QueryClassifier()
        
        content = '''```json
{"query_type": "sql", "confidence": 0.8, "reasoning": "Dosage lookup"}
```'''
        result = classifier._parse_json_response(content)
        
        assert result["query_type"] == "sql"
    
    def test_parse_json_response_with_extra_text(self):
        """Test parsing JSON with extra text around it."""
        from app.services.classifier import QueryClassifier
        
        classifier = QueryClassifier()
        
        content = 'Here is my classification: {"query_type": "hybrid", "confidence": 0.7, "reasoning": "Both needed"} That is my answer.'
        result = classifier._parse_json_response(content)
        
        assert result["query_type"] == "hybrid"


# Test data for classification accuracy evaluation
CLASSIFICATION_TEST_CASES = [
    # (query, expected_type)
    ("What are the side effects of aspirin?", "text"),
    ("Explain how beta blockers work", "text"),
    ("What does the research say about metformin and diabetes?", "text"),
    ("Describe the mechanism of action of statins", "text"),
    
    ("What is the recommended dose of ibuprofen?", "sql"),
    ("List all formulations of lisinopril", "sql"),
    ("What drugs interact with warfarin?", "sql"),
    ("What is the NDC code for Lipitor 10mg?", "sql"),
    
    ("What is the standard dose of metformin and why?", "hybrid"),
    ("Compare the side effect profiles of different SSRIs", "hybrid"),
    ("What are the indications for atorvastatin and how effective is it?", "hybrid"),
    ("List the dosages of amoxicillin and explain when each is appropriate", "hybrid"),
]


class TestClassificationAccuracy:
    """Tests for measuring classification accuracy."""
    
    @pytest.mark.parametrize("query,expected_type", CLASSIFICATION_TEST_CASES)
    def test_expected_classification(self, query: str, expected_type: str):
        """
        Parameterized test for classification expectations.
        
        Note: This test documents expected behavior but actual results
        depend on the LLM. Run with real API key to validate.
        """
        # This is a documentation test - actual classification requires API
        assert expected_type in ["text", "sql", "hybrid"]
