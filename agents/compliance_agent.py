"""
Compliance Agent for FinSage

This agent is responsible for verifying that financial advice and actions
comply with relevant regulations and best practices.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama

from utils.mcp_utils import ContextWrapper, get_registry
from contexts.user_profile_context import UserProfileContent, RiskTolerance
from contexts.portfolio_context import PortfolioContextContent

# Configure logger
logger = logging.getLogger(__name__)

class ComplianceAgent:
    """Agent for ensuring compliance with financial regulations"""
    
    def __init__(self):
        """Initialize the compliance agent"""
        self.agent_name = "ComplianceAgent"
        self.model = ChatOllama(model="gemma3:4b")
        
        # Set up tools for the agent
        self.tools = self._create_tools()
        
        # Set up the agent executor
        self.agent_executor = self._create_agent_executor()
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with tools and prompt"""
        # Set up the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial compliance expert responsible for ensuring that financial advice 
            and portfolio recommendations comply with regulations and best practices.
            
            Your role is to review financial advice, portfolio changes, and user interactions to:
            1. Identify potential compliance issues
            2. Provide disclaimers and warnings when necessary
            3. Validate that recommendations are appropriate for the user's risk profile
            4. Ensure proper disclosures are made about risks and limitations
            
            Always be thorough and conservative in your compliance checks.
            Format your responses in markdown for better readability.
            """  
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the agent
        agent = {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            )
        } | prompt | self.model | OpenAIFunctionsAgentOutputParser()
        
        # Create the agent executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def run(self, text_to_review: str, context_type: str = "advice") -> str:
        """
        Run the compliance check on the provided text
        
        Args:
            text_to_review: The text content to review for compliance
            context_type: The type of content being reviewed (advice, portfolio, etc.)
        """
        try:
            response = self.agent_executor.invoke({
                "input": f"Please review the following {context_type} for compliance issues:\n\n{text_to_review}"
            })
            return response["output"]
        except Exception as e:
            logger.error(f"Error running compliance agent: {e}")
            return f"Error performing compliance check: {str(e)}"
    
    def _create_tools(self) -> List[BaseTool]:
        """Create tools for the agent to use"""
        tools = []
        
        @tool("check_risk_suitability")
        def check_risk_suitability(advice_text: str) -> str:
            """
            Check if the financial advice is suitable for the user's risk profile.
            
            Args:
                advice_text: The financial advice text to check
            """
            # Get user profile from context registry
            registry = get_registry()
            user_profile_context = registry.get_latest_context("user_profile_context")
            
            if not user_profile_context:
                return "Cannot check risk suitability: No user profile found in context registry."
            
            user_profile = user_profile_context.content
            risk_tolerance = user_profile.risk_tolerance
            
            # Extract risk-related keywords from advice
            high_risk_keywords = [
                "aggressive", "high growth", "speculative", "crypto", 
                "volatile", "emerging markets", "leveraged", "options trading",
                "day trading", "futures", "margin", "high risk"
            ]
            
            moderate_risk_keywords = [
                "balanced", "moderate growth", "growth", "stocks", "equities",
                "moderate risk", "mutual funds", "index funds"
            ]
            
            conservative_risk_keywords = [
                "conservative", "income", "bonds", "fixed income", "treasuries",
                "low risk", "capital preservation", "low volatility", "defensive"
            ]
            
            # Count occurrences of keywords
            high_risk_count = sum(advice_text.lower().count(kw) for kw in high_risk_keywords)
            moderate_risk_count = sum(advice_text.lower().count(kw) for kw in moderate_risk_keywords)
            conservative_risk_count = sum(advice_text.lower().count(kw) for kw in conservative_risk_keywords)
            
            # Determine the risk level of the advice
            advice_risk_level = "UNKNOWN"
            if high_risk_count > moderate_risk_count and high_risk_count > conservative_risk_count:
                advice_risk_level = "AGGRESSIVE"
            elif moderate_risk_count > high_risk_count and moderate_risk_count > conservative_risk_count:
                advice_risk_level = "MODERATE"
            elif conservative_risk_count > high_risk_count and conservative_risk_count > moderate_risk_count:
                advice_risk_level = "CONSERVATIVE"
            
            # Check if the advice risk level matches the user's risk tolerance
            result = ["## Risk Suitability Analysis", ""]
            result.append(f"User Risk Tolerance: **{risk_tolerance.value.upper()}**")
            result.append(f"Advice Risk Level: **{advice_risk_level}**")
            
            if advice_risk_level == "UNKNOWN":
                result.append("\n**Risk Assessment:** Unable to determine the risk level of the advice.")
                result.append("Consider adding more specific risk-related information.")
            elif advice_risk_level == "AGGRESSIVE" and risk_tolerance == RiskTolerance.CONSERVATIVE:
                result.append("\n⚠️ **COMPLIANCE ISSUE:** The advice suggests high-risk investments that exceed the user's conservative risk tolerance.")
                result.append("Recommendation: Modify the advice to suggest more conservative investments or add strong risk warnings.")
            elif advice_risk_level == "MODERATE" and risk_tolerance == RiskTolerance.CONSERVATIVE:
                result.append("\n⚠️ **POTENTIAL COMPLIANCE ISSUE:** The advice suggests moderate-risk investments for a conservative investor.")
                result.append("Recommendation: Consider adjusting the advice or adding disclaimers about the increased risk level.")
            elif advice_risk_level == "AGGRESSIVE" and risk_tolerance == RiskTolerance.MODERATE:
                result.append("\n⚠️ **POTENTIAL COMPLIANCE ISSUE:** The advice suggests aggressive investments for a moderate-risk investor.")
                result.append("Recommendation: Consider reducing the risk level or adding strong disclaimers.")
            elif advice_risk_level == "CONSERVATIVE" and risk_tolerance == RiskTolerance.AGGRESSIVE:
                result.append("\n**Note:** The advice is more conservative than the user's risk tolerance.")
                result.append("This is not a compliance issue, but the user may prefer more growth-oriented options.")
            else:
                result.append("\n✓ **COMPLIANT:** The risk level of the advice appears suitable for the user's risk tolerance.")
            
            return "\n".join(result)
        
        @tool("check_disclosure_requirements")
        def check_disclosure_requirements(advice_text: str) -> str:
            """
            Check if the advice includes necessary disclosures and disclaimers.
            
            Args:
                advice_text: The financial advice text to check
            """
            # Required disclosure types
            disclosure_types = {
                "general_disclaimer": {
                    "required": True,
                    "patterns": [
                        r"not financial advice",
                        r"consult.*\s(advisor|professional)",
                        r"not guarantee",
                        r"past performance.*\s(not|no).*\s(indicative|guarantee)",
                        r"disclaimer",
                        r"subject to change",
                    ],
                    "description": "General disclaimer about the advice not being guaranteed"
                },
                "risk_warning": {
                    "required": True,
                    "patterns": [
                        r"risk",
                        r"may lose",
                        r"no guarantee",
                        r"volatile",
                        r"fluctuate",
                    ],
                    "description": "Warning about potential risks of investments"
                },
                "fee_disclosure": {
                    "required": False,  # Only required if specific products are recommended
                    "patterns": [
                        r"fee",
                        r"expense ratio",
                        r"commission",
                        r"charge",
                        r"cost",
                    ],
                    "description": "Disclosure of fees and expenses"
                },
                "tax_implications": {
                    "required": False,  # Only required if tax advice is given
                    "patterns": [
                        r"tax",
                        r"consult.*\stax professional",
                    ],
                    "description": "Disclosure about tax implications"
                }
            }
            
            # Check if tax-related advice is given
            tax_related_patterns = [
                r"tax",
                r"deduct",
                r"ira",
                r"401k",
                r"capital gain",
                r"tax-advantaged",
                r"tax-deferred",
            ]
            
            has_tax_advice = any(re.search(pattern, advice_text.lower()) for pattern in tax_related_patterns)
            if has_tax_advice:
                disclosure_types["tax_implications"]["required"] = True
            
            # Check if specific product recommendations are given
            product_patterns = [
                r"fund",
                r"etf",
                r"stock",
                r"bond",
                r"investment product",
                r"specific.*\sinvestment",
            ]
            
            has_product_recommendations = any(re.search(pattern, advice_text.lower()) for pattern in product_patterns)
            if has_product_recommendations:
                disclosure_types["fee_disclosure"]["required"] = True
            
            # Check for each disclosure type
            results = ["## Disclosure Requirements Analysis", ""]
            missing_disclosures = []
            
            for disc_type, disc_info in disclosure_types.items():
                if disc_info["required"]:
                    has_disclosure = any(re.search(pattern, advice_text.lower()) for pattern in disc_info["patterns"])
                    
                    if has_disclosure:
                        results.append(f"✓ **{disc_type.replace('_', ' ').title()}**: Found")
                    else:
                        results.append(f"❌ **{disc_type.replace('_', ' ').title()}**: Missing")
                        missing_disclosures.append(disc_info["description"])
            
            # Summary
            if missing_disclosures:
                results.append("\n### Missing Required Disclosures")
                for i, missing in enumerate(missing_disclosures, 1):
                    results.append(f"{i}. {missing}")
                
                results.append("\n⚠️ **COMPLIANCE ISSUE:** The advice is missing required disclosures.")
                results.append("Please add the necessary disclaimers and disclosures listed above.")
            else:
                results.append("\n✓ **COMPLIANT:** All required disclosures appear to be present.")
            
            # Example disclaimer templates
            if missing_disclosures:
                results.append("\n### Suggested Disclaimer Templates")
                
                if "General disclaimer about the advice not being guaranteed" in missing_disclosures:
                    results.append("""
**General Disclaimer Template:**
*This information is for educational purposes only and is not financial advice. 
Consult with a professional financial advisor before making any investment decisions. 
Past performance is not indicative of future results.*
""")
                
                if "Warning about potential risks of investments" in missing_disclosures:
                    results.append("""
**Risk Warning Template:**
*All investments involve risk and may lose value. The value of your investment can go down depending on market conditions. 
Different investment types have varying levels of risk.*
""")
                
                if "Disclosure of fees and expenses" in missing_disclosures and has_product_recommendations:
                    results.append("""
**Fee Disclosure Template:**
*Investments may be subject to management fees, transaction costs, and other expenses that can impact returns. 
Be sure to understand all fees associated with any investment before proceeding.*
""")
                
                if "Disclosure about tax implications" in missing_disclosures and has_tax_advice:
                    results.append("""
**Tax Implications Template:**
*This information is not tax advice. Tax laws are complex and subject to change. 
Consult with a tax professional regarding your specific situation before making tax-related financial decisions.*
""")
            
            return "\n".join(results)
        
        @tool("verify_factual_accuracy")
        def verify_factual_accuracy(advice_text: str) -> str:
            """
            Check the advice for potential factual inaccuracies or misleading information.
            
            Args:
                advice_text: The financial advice text to check
            """
            # Common factual issues to check for
            factual_checks = [
                {
                    "pattern": r"guarantee.*\s(return|profit|income)",
                    "issue": "Claims of guaranteed returns",
                    "correction": "Investment returns cannot be guaranteed"
                },
                {
                    "pattern": r"(always|never|certainly|definitely|guaranteed to)\s(increase|decrease|rise|fall|grow)",
                    "issue": "Absolute predictions about market movements",
                    "correction": "Market movements cannot be predicted with certainty"
                },
                {
                    "pattern": r"(best|worst|only|perfect)\s(investment|strategy|approach|option)",
                    "issue": "Superlative claims without qualification",
                    "correction": "Investment suitability is individual and no single option is universally 'best'"
                },
                {
                    "pattern": r"(double|triple|[0-9]+%)\s(return|gain|profit)",
                    "issue": "Specific return projections",
                    "correction": "Specific return percentages should not be promised"
                },
                {
                    "pattern": r"(no|zero)\s(risk|downside)",
                    "issue": "Claims of no risk",
                    "correction": "All investments carry some level of risk"
                },
                {
                    "pattern": r"(tax.free|no.tax)",
                    "issue": "Absolute tax claims",
                    "correction": "Tax outcomes depend on individual circumstances and tax laws"
                },
                {
                    "pattern": r"(everyone|anybody|all investors)\s(should|must|needs to)",
                    "issue": "Universal recommendations",
                    "correction": "Recommendations should be tailored to individual circumstances"
                }
            ]
            
            # Check for each factual issue
            results = ["## Factual Accuracy Analysis", ""]
            identified_issues = []
            
            for check in factual_checks:
                matches = re.finditer(check["pattern"], advice_text.lower())
                for match in matches:
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(advice_text), match.end() + 50)
                    context = "..." + advice_text[context_start:context_end] + "..."
                    
                    identified_issues.append({
                        "issue": check["issue"],
                        "context": context,
                        "correction": check["correction"]
                    })
            
            # Report results
            if identified_issues:
                results.append("⚠️ **POTENTIAL FACTUAL ISSUES DETECTED**\n")
                
                for i, issue in enumerate(identified_issues, 1):
                    results.append(f"**Issue {i}: {issue['issue']}**")
                    results.append(f"Context: \"{issue['context']}\"")
                    results.append(f"Correction: {issue['correction']}\n")
                
                results.append("**COMPLIANCE RECOMMENDATION:**")
                results.append("Revise the statements above to avoid making absolute claims, unrealistic promises, or misleading statements.")
            else:
                results.append("✓ **No common factual issues detected.**")
                results.append("The advice appears to avoid making unrealistic promises or misleading claims.")
            
            # Additional suggestions for maintaining factual accuracy
            results.append("\n### Best Practices for Factual Accuracy")
            results.append("1. Use qualifying language like 'may', 'could', or 'historically has' instead of absolutes")
            results.append("2. Provide context for historical performance data")
            results.append("3. Clarify that recommendations are based on current market conditions")
            results.append("4. Disclose the limitations of any analysis or projection")
            
            return "\n".join(results)
        
        @tool("generate_compliance_summary")
        def generate_compliance_summary(advice_text: str) -> str:
            """
            Generate a comprehensive compliance summary for the given financial advice.
            This tool combines multiple compliance checks into a single report.
            
            Args:
                advice_text: The financial advice text to check
            """
            # Run all compliance checks
            risk_result = check_risk_suitability(advice_text)
            disclosure_result = check_disclosure_requirements(advice_text)
            factual_result = verify_factual_accuracy(advice_text)
            
            # Determine overall compliance status
            has_compliance_issues = "COMPLIANCE ISSUE" in risk_result or "COMPLIANCE ISSUE" in disclosure_result or "FACTUAL ISSUES DETECTED" in factual_result
            
            # Generate summary report
            summary = ["# Compliance Review Summary", ""]
            
            if has_compliance_issues:
                summary.append("⚠️ **COMPLIANCE ISSUES DETECTED**")
                summary.append("The following issues must be addressed before sharing this advice:")
            else:
                summary.append("✅ **COMPLIANT**")
                summary.append("No major compliance issues detected. Minor suggestions may still apply.")
            
            # Add executive summary of issues
            if "COMPLIANCE ISSUE" in risk_result:
                summary.append("- **Risk Suitability:** Issues found - Advice risk level may not match user's risk tolerance")
            else:
                summary.append("- **Risk Suitability:** Compliant")
                
            if "COMPLIANCE ISSUE" in disclosure_result:
                summary.append("- **Required Disclosures:** Issues found - Missing required disclaimers")
            else:
                summary.append("- **Required Disclosures:** Compliant")
                
            if "FACTUAL ISSUES DETECTED" in factual_result:
                summary.append("- **Factual Accuracy:** Issues found - Potentially misleading statements detected")
            else:
                summary.append("- **Factual Accuracy:** Compliant")
            
            # Add detailed reports
            summary.append("\n## Detailed Compliance Reports\n")
            summary.append("### 1. Risk Suitability Assessment")
            summary.extend(risk_result.split("\n")[1:])  # Skip the title line
            
            summary.append("\n### 2. Disclosure Requirements")
            summary.extend(disclosure_result.split("\n")[1:])  # Skip the title line
            
            summary.append("\n### 3. Factual Accuracy")
            summary.extend(factual_result.split("\n")[1:])  # Skip the title line
            
            # Final recommendations
            summary.append("\n## Final Compliance Recommendations")
            
            if has_compliance_issues:
                summary.append("1. Address all identified compliance issues")
                summary.append("2. Re-run compliance check after making corrections")
                summary.append("3. Document all compliance reviews for record-keeping purposes")
            else:
                summary.append("1. Maintain records of this compliance review")
                summary.append("2. Consider implementing suggested best practices where applicable")
                summary.append("3. Re-review if significant changes are made to the advice")
            
            summary.append("\n*This compliance review was generated automatically and should be reviewed by a compliance professional before final approval.*")
            
            return "\n".join(summary)
        
        # Add all tools to the list
        tools.append(check_risk_suitability)
        tools.append(check_disclosure_requirements)
        tools.append(verify_factual_accuracy)
        tools.append(generate_compliance_summary)
        
        return tools
