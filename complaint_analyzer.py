"""
NLP-based Complaint Severity Analyzer
Analyzes complaint text to determine severity scores for better risk assessment
"""

import re
import numpy as np
from typing import Tuple, Dict, List


class ComplaintSeverityAnalyzer:
    """Analyzes complaint text to determine severity scores"""
    
    def __init__(self):
        """Initialize the complaint analyzer with severity keywords"""
        
        # Severity keywords and their impact multipliers
        self.severity_keywords = {
            # Extreme severity (3.0x multiplier)
            'extreme': ['catastrophic', 'massive', 'severe', 'epidemic', 'outbreak', 'emergency'],
            
            # High severity (2.5x multiplier)
            'high': ['flooding', 'stagnant', 'contaminated', 'continuous', 'persistent', 'widespread',
                     'critical', 'urgent', 'widespread', 'breeding'],
            
            # Medium-high severity (2.0x multiplier)
            'medium_high': ['clogged', 'overflowing', 'water', 'mosquito', 'blockage', 'accumulation',
                           'pipe', 'drain', 'poor', 'conditions'],
            
            # Medium severity (1.5x multiplier)
            'medium': ['report', 'complaint', 'issue', 'problem', 'concern', 'incident'],
        }
        
        # Duration indicators (increases multiplier based on days/duration)
        self.duration_patterns = {
            r'(\d+)\s*(?:days?|weeks?|months?|years?)': 'long_duration',
            r'(?:for|over|since)\s*(\d+)': 'long_duration',
            r'(?:continuous|persistent|ongoing|repeated)': 'ongoing',
        }
        
        # Location multipliers (specific areas of concern)
        self.location_keywords = {
            'slum': 1.3,
            'commercial': 1.2,
            'market': 1.3,
            'hospital': 1.4,
            'school': 1.2,
            'residential': 1.0,
        }
        
        # Impact indicators
        self.impact_keywords = {
            'health': 1.4,
            'disease': 1.5,
            'illness': 1.3,
            'infection': 1.4,
            'cases': 1.5,
            'fever': 1.3,
            'sick': 1.2,
        }
    
    def analyze_complaint_text(self, complaint_text: str) -> Tuple[float, Dict]:
        """
        Analyze complaint text and return severity score
        
        Args:
            complaint_text: Text description of the complaint
            
        Returns:
            Tuple of (severity_score, analysis_dict)
            severity_score: 0-10 scale (0 = least severe, 10 = most severe)
            analysis_dict: Breakdown of analysis components
        """
        
        if not complaint_text or len(complaint_text.strip()) == 0:
            return 0.0, {
                'severity_score': 0.0,
                'base_score': 0,
                'keywords_found': [],
                'duration_factor': 1.0,
                'location_factor': 1.0,
                'impact_factor': 1.0,
                'final_multiplier': 1.0,
                'text_length_factor': 1.0
            }
        
        text = complaint_text.lower().strip()
        
        # 1. Base score from keyword detection
        base_score = self._get_base_score(text)
        keywords_found = self._detect_keywords(text)
        
        # 2. Duration multiplier
        duration_factor = self._calculate_duration_factor(text)
        
        # 3. Location multiplier
        location_factor = self._detect_location_factor(text)
        
        # 4. Impact multiplier (health-related)
        impact_factor = self._detect_impact_factor(text)
        
        # 5. Text length factor (longer, detailed complaints are usually more serious)
        text_length_factor = self._calculate_text_length_factor(text)
        
        # 6. Combine all factors
        final_multiplier = (duration_factor * location_factor * 
                          impact_factor * text_length_factor)
        
        # Calculate final severity score (0-10 scale)
        severity_score = min(10.0, base_score * final_multiplier)
        
        analysis = {
            'severity_score': round(severity_score, 2),
            'base_score': round(base_score, 2),
            'keywords_found': keywords_found,
            'duration_factor': round(duration_factor, 2),
            'location_factor': round(location_factor, 2),
            'impact_factor': round(impact_factor, 2),
            'text_length_factor': round(text_length_factor, 2),
            'final_multiplier': round(final_multiplier, 2)
        }
        
        return severity_score, analysis
    
    def calculate_weighted_complaint_score(self, complaint_count: float, 
                                          complaint_text: str = "") -> float:
        """
        Calculate weighted complaint score combining count and severity
        
        Args:
            complaint_count: Raw count of complaints
            complaint_text: Optional text description of complaint(s)
            
        Returns:
            Weighted complaint score
        """
        
        # If no text provided, use base count
        if not complaint_text:
            return float(complaint_count)
        
        # Get severity multiplier from text analysis
        severity_score, _ = self.analyze_complaint_text(complaint_text)
        
        # Normalize severity score to multiplier (0-10 -> 1-2x)
        severity_multiplier = 1.0 + (severity_score / 10.0)
        
        # Calculate weighted score
        weighted_score = complaint_count * severity_multiplier
        
        return round(weighted_score, 2)
    
    def _get_base_score(self, text: str) -> float:
        """Calculate base score from severity keyword categories"""
        
        base_scores = {
            'extreme': 8.0,
            'high': 6.0,
            'medium_high': 4.0,
            'medium': 2.0,
        }
        
        max_score = 0.0
        for category, keywords in self.severity_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    score = base_scores.get(category, 1.0)
                    max_score = max(max_score, score)
        
        # If no keywords found, give a small base score
        if max_score == 0.0:
            max_score = 1.0 if len(text.split()) >= 5 else 0.5
        
        return max_score
    
    def _detect_keywords(self, text: str) -> List[str]:
        """Detect and return all severity keywords found"""
        
        found_keywords = []
        for category, keywords in self.severity_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    found_keywords.append(f"{keyword}({category})")
        
        return found_keywords[:10]  # Return top 10 found keywords
    
    def _calculate_duration_factor(self, text: str) -> float:
        """Calculate multiplier based on duration mentioned in complaint"""
        
        duration_factor = 1.0
        
        # Check for long duration patterns
        for pattern in self.duration_patterns.keys():
            matches = re.finditer(pattern, text)
            for match in matches:
                if 'week' in pattern or 'month' in pattern or 'year' in pattern:
                    duration_factor = min(duration_factor + 0.5, 2.5)
                elif 'day' in pattern:
                    try:
                        days = int(match.group(1))
                        if days >= 5:
                            duration_factor = 2.0
                        elif days >= 3:
                            duration_factor = 1.5
                    except (ValueError, IndexError):
                        pass
        
        # Check for ongoing/continuous indicators
        if 'continuous' in text or 'persistent' in text or 'ongoing' in text:
            duration_factor = max(duration_factor, 2.0)
        
        return duration_factor
    
    def _detect_location_factor(self, text: str) -> float:
        """Calculate multiplier based on location mentioned"""
        
        location_factor = 1.0
        for location, multiplier in self.location_keywords.items():
            if location in text:
                location_factor = max(location_factor, multiplier)
        
        return location_factor
    
    def _detect_impact_factor(self, text: str) -> float:
        """Calculate multiplier based on health/disease impact"""
        
        impact_factor = 1.0
        impact_count = 0
        
        for impact_word, multiplier in self.impact_keywords.items():
            if impact_word in text:
                impact_count += 1
                impact_factor = max(impact_factor, multiplier)
        
        # Additional boost if multiple impact keywords
        if impact_count >= 2:
            impact_factor = min(impact_factor + 0.2, 1.6)
        
        return impact_factor
    
    def _calculate_text_length_factor(self, text: str) -> float:
        """Calculate factor based on detail level (text length)"""
        
        word_count = len(text.split())
        
        if word_count < 5:
            return 1.0
        elif word_count < 15:
            return 1.1
        elif word_count < 30:
            return 1.2
        else:
            return 1.3  # Longer, more detailed complaints
    
    def batch_analyze_complaints(self, complaints_list: List[Dict]) -> List[Dict]:
        """
        Analyze multiple complaints
        
        Args:
            complaints_list: List of dicts with 'count' and 'text' keys
            
        Returns:
            List of analysis results with weighted scores
        """
        
        results = []
        for complaint in complaints_list:
            count = complaint.get('count', 0)
            text = complaint.get('text', '')
            
            severity_score, analysis = self.analyze_complaint_text(text)
            weighted_score = self.calculate_weighted_complaint_score(count, text)
            
            results.append({
                'original_count': count,
                'complaint_text': text,
                'severity_score': severity_score,
                'weighted_score': weighted_score,
                'analysis': analysis
            })
        
        return results


# Example usage and testing
if __name__ == "__main__":
    analyzer = ComplaintSeverityAnalyzer()
    
    # Test complaints
    test_complaints = [
        "No complaint",
        "5 reports of garbage",
        "Massive flooding and stagnant water for 5 days in slum area",
        "Persistent waterlogging causing health issues and disease spread",
        "Emergency: Contaminated water in school causing dengue cases",
    ]
    
    print("=" * 80)
    print("COMPLAINT SEVERITY ANALYSIS")
    print("=" * 80)
    
    for complaint in test_complaints:
        severity, analysis = analyzer.analyze_complaint_text(complaint)
        print(f"\nComplaint: {complaint}")
        print(f"Severity Score: {severity}/10")
        print(f"Analysis: {analysis}")
        print("-" * 80)
    
    # Test weighted scoring
    print("\n" + "=" * 80)
    print("WEIGHTED COMPLAINT SCORING")
    print("=" * 80)
    
    test_cases = [
        (10, ""),  # Just count
        (10, "Minor drainage issue"),
        (10, "Massive flooding for 5 days causing health crisis"),
    ]
    
    for count, text in test_cases:
        weighted = analyzer.calculate_weighted_complaint_score(count, text)
        print(f"\nCount: {count}, Text: '{text}'")
        print(f"Weighted Score: {weighted}")
