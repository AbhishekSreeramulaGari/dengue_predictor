"""
AI Assistant for Dengue Prevention and Management
Provides contextual advice for public users and administrative recommendations
"""

import random
from typing import List, Dict, Any
from datetime import datetime


class DengueAIAssistant:
    """AI Assistant for dengue-related queries and recommendations"""

    def __init__(self):
        """Initialize the AI assistant with knowledge base"""
        self.public_knowledge = {
            'prevention': [
                "Eliminate standing water in containers, gutters, and around your home every week",
                "Use mosquito repellents containing DEET, picaridin, or oil of lemon eucalyptus",
                "Wear long-sleeved shirts and long pants when outdoors, especially during dawn and dusk",
                "Install or repair window and door screens to keep mosquitoes outside",
                "Keep your surroundings clean and dispose of garbage properly in covered bins"
            ],
            'symptoms': [
                "High fever (104°F/40°C) that may come and go",
                "Severe headache, often behind the eyes",
                "Pain behind the eyes, in muscles and joints",
                "Nausea and vomiting",
                "Rash that appears 2-5 days after fever starts",
                "Mild bleeding from nose or gums"
            ],
            'sanitation': [
                "Ensure proper drainage systems to prevent water stagnation",
                "Clean water storage containers weekly with chlorine",
                "Remove discarded tires, bottles, and containers that can collect water",
                "Maintain clean gutters and downspouts",
                "Report garbage accumulation and waterlogging issues immediately"
            ],
            'seasonal': [
                "During monsoon season, check for water accumulation daily",
                "Use bed nets treated with insecticide during high-risk periods",
                "Avoid outdoor activities during peak mosquito hours (dawn and dusk)",
                "Keep windows closed or use mosquito nets during rainy season",
                "Participate in community clean-up drives"
            ]
        }

        self.admin_knowledge = {
            'outbreak_response': [
                "Implement immediate vector control measures in affected wards",
                "Set up medical surveillance and rapid response teams",
                "Conduct mass fogging operations in high-risk areas",
                "Establish temporary medical camps for fever surveillance",
                "Coordinate with local health authorities for vaccine distribution if available"
            ],
            'ward_management': [
                "Prioritize wards with high complaint density for intervention",
                "Implement targeted larvicide application in breeding hotspots",
                "Conduct house-to-house surveys to identify hidden breeding sites",
                "Set up community awareness programs with local volunteers",
                "Monitor weather patterns to predict high-risk periods"
            ],
            'resource_allocation': [
                "Deploy fogging machines and larvicides to high-risk wards first",
                "Train local health workers for fever surveillance",
                "Establish communication channels with community leaders",
                "Set up emergency supply chains for medical equipment",
                "Create ward-wise action plans based on risk assessment"
            ],
            'monitoring': [
                "Track daily case numbers and hospitalization rates",
                "Monitor rainfall patterns and temperature changes",
                "Analyze complaint trends to identify emerging hotspots",
                "Conduct regular entomological surveys for mosquito density",
                "Evaluate the effectiveness of control measures weekly"
            ]
        }

    def get_public_assistance(self, query: str = "", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Provide assistance for public users

        Args:
            query: User's question or request
            context: Additional context (ward, risk level, etc.)

        Returns:
            Dictionary with response and recommendations
        """

        query_lower = query.lower() if query else ""

        # Determine response type based on query
        if any(word in query_lower for word in ['prevent', 'avoid', 'stop', 'protection']):
            response_type = 'prevention'
            title = "Dengue Prevention Tips"
        elif any(word in query_lower for word in ['symptom', 'sign', 'feel', 'sick']):
            response_type = 'symptoms'
            title = "Dengue Symptoms"
        elif any(word in query_lower for word in ['clean', 'garbage', 'water', 'sanitation']):
            response_type = 'sanitation'
            title = "Sanitation and Cleanliness"
        elif any(word in query_lower for word in ['season', 'rain', 'monsoon', 'weather']):
            response_type = 'seasonal'
            title = "Seasonal Precautions"
        else:
            response_type = 'general'
            title = "General Dengue Information"

        # Get relevant information
        if response_type == 'general':
            recommendations = random.sample(self.public_knowledge['prevention'], 3)
            additional_info = random.sample(self.public_knowledge['symptoms'], 2)
            recommendations.extend(additional_info)
        else:
            recommendations = self.public_knowledge.get(response_type, self.public_knowledge['prevention'])

        # Add context-specific advice
        contextual_advice = []
        if context:
            risk_level = context.get('risk_level', '').lower()
            if 'high' in risk_level or 'very high' in risk_level:
                contextual_advice.append("⚠️ High risk period: Take extra precautions and monitor for symptoms closely")
            elif 'medium' in risk_level:
                contextual_advice.append("⚡ Medium risk: Increase preventive measures and stay vigilant")

            ward_name = context.get('ward_name')
            if ward_name:
                contextual_advice.append(f"📍 Location-specific: Focus on reported issues in {ward_name}")

        return {
            'title': title,
            'recommendations': recommendations,
            'contextual_advice': contextual_advice,
            'emergency_contact': "If you experience dengue symptoms, seek medical attention immediately. Call emergency services or visit nearest hospital.",
            'source': 'Dengue Prevention Guidelines'
        }

    def get_admin_assistance(self, query: str = "", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Provide assistance for admin/authority users

        Args:
            query: Admin's question or request
            context: Additional context (complaints, ward data, etc.)

        Returns:
            Dictionary with response and recommendations
        """

        query_lower = query.lower() if query else ""

        # Determine response type based on query
        if any(word in query_lower for word in ['outbreak', 'emergency', 'crisis', 'response']):
            response_type = 'outbreak_response'
            title = "Outbreak Response Strategies"
        elif any(word in query_lower for word in ['ward', 'area', 'location', 'zone']):
            response_type = 'ward_management'
            title = "Ward-Level Management"
        elif any(word in query_lower for word in ['resource', 'allocation', 'supply', 'equipment']):
            response_type = 'resource_allocation'
            title = "Resource Allocation Guidelines"
        elif any(word in query_lower for word in ['monitor', 'track', 'surveil', 'data']):
            response_type = 'monitoring'
            title = "Monitoring and Surveillance"
        else:
            response_type = 'general'
            title = "Administrative Recommendations"

        # Get relevant information
        if response_type == 'general':
            recommendations = []
            for category in ['ward_management', 'monitoring', 'resource_allocation']:
                recommendations.extend(random.sample(self.admin_knowledge[category], 2))
        else:
            recommendations = self.admin_knowledge.get(response_type, self.admin_knowledge['ward_management'])

        # Add context-specific advice
        contextual_advice = []
        if context:
            complaint_count = context.get('complaint_count', 0)
            if complaint_count > 50:
                contextual_advice.append("🚨 High complaint volume: Prioritize immediate intervention in affected wards")
            elif complaint_count > 20:
                contextual_advice.append("⚠️ Moderate complaint load: Monitor closely and prepare response teams")

            high_risk_wards = context.get('high_risk_wards', [])
            if high_risk_wards:
                ward_names = [str(w) for w in high_risk_wards[:3]]
                contextual_advice.append(f"🎯 Focus areas: {', '.join(ward_names)} require immediate attention")

        return {
            'title': title,
            'recommendations': recommendations,
            'contextual_advice': contextual_advice,
            'action_items': [
                "Review current complaint status and prioritize critical issues",
                "Coordinate with health department for medical support",
                "Update community through local communication channels",
                "Monitor intervention effectiveness and adjust strategies"
            ],
            'source': 'Municipal Health Authority Guidelines'
        }

    def get_ward_specific_advice(self, ward_name: str, risk_level: str) -> Dict[str, Any]:
        """
        Get ward-specific advice based on risk level

        Args:
            ward_name: Name of the ward
            risk_level: Current risk level

        Returns:
            Ward-specific recommendations
        """

        base_advice = {
            'Low Risk': [
                f"Maintain regular monitoring in {ward_name}",
                "Continue community education programs",
                "Ensure proper waste management systems"
            ],
            'Medium Risk': [
                f"Implement targeted interventions in {ward_name}",
                "Increase surveillance and reporting",
                "Conduct awareness campaigns in local communities"
            ],
            'High Risk': [
                f"Deploy emergency response teams to {ward_name}",
                "Implement intensive vector control measures",
                "Set up medical monitoring stations"
            ],
            'Very High Risk': [
                f"Declare {ward_name} as critical zone",
                "Implement round-the-clock monitoring",
                "Coordinate with state health authorities"
            ]
        }

        recommendations = base_advice.get(risk_level, base_advice['Medium Risk'])

        return {
            'ward': ward_name,
            'risk_level': risk_level,
            'recommendations': recommendations,
            'timeline': 'Immediate action required within 24-48 hours',
            'coordination': 'Coordinate with local ward officers and health workers'
        }