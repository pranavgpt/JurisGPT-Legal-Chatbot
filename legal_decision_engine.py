"""
Legal Decision Engine Module
This module provides structured legal analysis and decision-making capabilities
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class IPCSection:
    number: str
    title: str
    punishment: str
    bailable: bool
    cognizable: bool
    elements: List[str]

@dataclass
class LegalDecision:
    applicable_sections: List[IPCSection]
    severity: str  # "Minor", "Major", "Heinous"
    bail_likelihood: str
    punishment_range: str
    legal_reasoning: str

class LegalDecisionEngine:
    """Core decision engine for legal case analysis"""
    
    def __init__(self):
        self.ipc_database = self._initialize_ipc_database()
        self.severity_mapping = {
            'heinous': ['302', '376', '364A', '396'],
            'major': ['420', '409', '406', '498A', '354'],
            'minor': ['323', '379', '294', '506']
        }
    
    def _initialize_ipc_database(self) -> Dict[str, IPCSection]:
        """Initialize comprehensive IPC database"""
        return {
            "302": IPCSection("302", "Murder", "Death or Life imprisonment", False, True, 
                             ["Intention to cause death", "Knowledge likely to cause death", "Death occurs"]),
            "376": IPCSection("376", "Rape", "Minimum 7 years imprisonment", False, True,
                             ["Sexual intercourse", "Against will/without consent", "With woman"]),
            "420": IPCSection("420", "Cheating", "Up to 7 years + fine", False, True,
                             ["Deception", "Dishonest inducement", "Property delivery"]),
            "379": IPCSection("379", "Theft", "Up to 3 years or fine", True, True,
                             ["Dishonest intention", "Movable property", "Taking without consent"]),
            "323": IPCSection("323", "Voluntarily causing hurt", "Up to 1 year or Rs. 1000", True, True,
                             ["Voluntary action", "Causing hurt", "No grievous hurt"]),
            "354": IPCSection("354", "Assault on woman with intent to outrage modesty", "Up to 2 years + fine", False, True,
                             ["Assault or criminal force", "Against woman", "Intent to outrage modesty"]),
            "498A": IPCSection("498A", "Cruelty by husband or relatives", "Up to 3 years + fine", False, True,
                              ["Cruelty to woman", "By husband/relatives", "Mental/physical harassment"]),
            "506": IPCSection("506", "Criminal intimidation", "Up to 2 years or fine", True, True,
                             ["Threatening", "Causing alarm", "Intent to intimidate"])
        }
    
    def extract_case_facts(self, case_text: str) -> Dict[str, List[str]]:
        """Extract structured facts from case description"""
        facts = {
            'actions': [],
            'parties': [],
            'objects': [],
            'circumstances': []
        }
        
        # Extract actions (verbs that might indicate crimes)
        action_patterns = [
            r'\b(killed|murdered|stabbed|shot)\b',
            r'\b(raped|molested|assaulted)\b',
            r'\b(stole|robbed|burgled|cheated)\b',
            r'\b(threatened|intimidated|blackmailed)\b',
            r'\b(hit|hurt|injured|beaten)\b'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, case_text, re.IGNORECASE)
            facts['actions'].extend(matches)
        
        return facts
    
    def identify_applicable_sections(self, case_text: str) -> List[IPCSection]:
        """Identify applicable IPC sections based on case facts"""
        applicable_sections = []
        case_lower = case_text.lower()
        
        # Section mapping based on keywords
        section_keywords = {
            "302": ["murder", "killed", "death", "homicide"],
            "376": ["rape", "sexual assault", "molest"],
            "420": ["cheat", "fraud", "deceive", "dishonest"],
            "379": ["theft", "steal", "rob"],
            "323": ["hurt", "assault", "beat", "hit"],
            "354": ["outrage", "modesty", "inappropriate touch"],
            "498A": ["dowry", "harassment", "cruelty", "husband"],
            "506": ["threat", "intimidate", "blackmail"]
        }
        
        for section_num, keywords in section_keywords.items():
            if any(keyword in case_lower for keyword in keywords):
                if section_num in self.ipc_database:
                    applicable_sections.append(self.ipc_database[section_num])
        
        return applicable_sections
    
    def determine_severity(self, sections: List[IPCSection]) -> str:
        """Determine case severity based on applicable sections"""
        section_numbers = [s.number for s in sections]
        
        for severity, severity_sections in self.severity_mapping.items():
            if any(section in section_numbers for section in severity_sections):
                return severity.title()
        
        return "Minor"
    
    def assess_bail_likelihood(self, sections: List[IPCSection], severity: str) -> str:
        """Assess likelihood of bail being granted"""
        non_bailable_count = sum(1 for s in sections if not s.bailable)
        
        if non_bailable_count > 0:
            if severity == "Heinous":
                return "Very Unlikely - Non-bailable offense with heinous nature"
            elif severity == "Major":
                return "Unlikely - Non-bailable offense, may require special circumstances"
            else:
                return "Possible - Non-bailable but may be considered based on circumstances"
        else:
            return "Likely - All applicable sections are bailable"
    
    def generate_punishment_range(self, sections: List[IPCSection]) -> str:
        """Generate expected punishment range"""
        if not sections:
            return "No applicable sections identified"
        
        punishments = [s.punishment for s in sections]
        
        # Extract maximum punishment
        max_punishment = "Fine or imprisonment as per applicable sections"
        
        if any("death" in p.lower() for p in punishments):
            max_punishment = "Death penalty or life imprisonment"
        elif any("life" in p.lower() for p in punishments):
            max_punishment = "Life imprisonment"
        elif any("7 years" in p for p in punishments):
            max_punishment = "Up to 7 years imprisonment"
        elif any("3 years" in p for p in punishments):
            max_punishment = "Up to 3 years imprisonment"
        
        return max_punishment
    
    def analyze_case(self, case_description: str) -> LegalDecision:
        """Main method to analyze a legal case and provide structured decision"""
        
        # Step 1: Identify applicable IPC sections
        applicable_sections = self.identify_applicable_sections(case_description)
        
        # Step 2: Determine case severity
        severity = self.determine_severity(applicable_sections)
        
        # Step 3: Assess bail likelihood
        bail_likelihood = self.assess_bail_likelihood(applicable_sections, severity)
        
        # Step 4: Generate punishment range
        punishment_range = self.generate_punishment_range(applicable_sections)
        
        # Step 5: Generate legal reasoning
        legal_reasoning = self._generate_legal_reasoning(case_description, applicable_sections, severity)
        
        return LegalDecision(
            applicable_sections=applicable_sections,
            severity=severity,
            bail_likelihood=bail_likelihood,
            punishment_range=punishment_range,
            legal_reasoning=legal_reasoning
        )
    
    def _generate_legal_reasoning(self, case_description: str, sections: List[IPCSection], severity: str) -> str:
        """Generate detailed legal reasoning for the decision"""
        reasoning_parts = []
        
        # Case analysis
        reasoning_parts.append(f"Based on the case facts, this appears to be a {severity.lower()} offense.")
        
        # Section-wise analysis
        if sections:
            reasoning_parts.append("The following IPC sections are applicable:")
            for section in sections:
                reasoning_parts.append(f"- Section {section.number} ({section.title}): {section.punishment}")
                reasoning_parts.append(f"  Essential elements: {', '.join(section.elements)}")
        
        # Legal precedent consideration
        reasoning_parts.append("\nLegal considerations:")
        reasoning_parts.append("- The prosecution must prove all essential elements beyond reasonable doubt")
        reasoning_parts.append("- The accused is entitled to legal representation and fair trial")
        reasoning_parts.append("- Mitigating and aggravating factors should be considered during sentencing")
        
        return "\n".join(reasoning_parts)

def format_legal_decision(decision: LegalDecision, case_description: str) -> str:
    """Format legal decision into structured output"""
    
    output = []
    output.append("=" * 60)
    output.append("LEGAL CASE ANALYSIS & DECISION")
    output.append("=" * 60)
    
    # Case Summary
    output.append("\n📋 CASE SUMMARY:")
    output.append(f"Case involves: {case_description[:200]}...")
    output.append(f"Severity Classification: {decision.severity}")
    
    # Applicable IPC Sections
    output.append("\n⚖️ APPLICABLE IPC SECTIONS:")
    if decision.applicable_sections:
        for section in decision.applicable_sections:
            output.append(f"• Section {section.number}: {section.title}")
            output.append(f"  Punishment: {section.punishment}")
            output.append(f"  Bailable: {'Yes' if section.bailable else 'No'}")
            output.append(f"  Elements: {', '.join(section.elements)}")
            output.append("")
    else:
        output.append("No specific IPC sections identified. Further legal research required.")
    
    # Legal Decision
    output.append("🏛️ DECISION & REASONING:")
    output.append(decision.legal_reasoning)
    
    # Bail Assessment
    output.append(f"\n🔒 BAIL LIKELIHOOD:")
    output.append(decision.bail_likelihood)
    
    # Punishment Range
    output.append(f"\n⚖️ LIKELY PUNISHMENT:")
    output.append(decision.punishment_range)
    
    # Recommendations
    output.append("\n💡 RECOMMENDATIONS:")
    output.append("1. Gather all evidence and witness statements")
    output.append("2. Ensure proper legal representation")
    output.append("3. Follow due process requirements")
    output.append("4. Consider plea bargaining if applicable")
    
    return "\n".join(output)

# Example usage and testing
if __name__ == "__main__":
    engine = LegalDecisionEngine()
    
    # Test cases
    test_cases = [
        "A person threatened someone with a knife and demanded money from them",
        "Someone was found to have cheated multiple people by promising false investment returns",
        "A person was assaulted and injured during a fight outside a bar",
        "A woman complained that her husband and in-laws were harassing her for dowry"
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"TEST CASE {i}")
        print(f"{'='*50}")
        
        decision = engine.analyze_case(case)
        formatted_output = format_legal_decision(decision, case)
        print(formatted_output)