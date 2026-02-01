# src/llm_handler.py - IMPROVED VERSION
import os
from typing import Dict, List
from dotenv import load_dotenv
import re

load_dotenv()

class LLMHandler:
    def __init__(self, use_openai: bool = False):
        self.use_openai = use_openai
        
        if use_openai:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.model = "gpt-3.5-turbo"
        else:
            self.model = "local"
    
    def generate_answer(self, context: str, question: str) -> Dict:
        """Generate answer with strict grounding rules"""
        
        if self.use_openai:
            return self._call_openai(context, question)
        else:
            return self._call_local_llm(context, question)
    
    def _call_openai(self, context: str, question: str) -> Dict:
        """Call OpenAI API"""
        import openai
        
        system_prompt = """You are a strict document-based AI assistant.

RULES:
1. Use ONLY the provided context.
2. If answer not found in context, respond exactly: "Answer not found in provided documents."
3. Do not assume anything not in context.
4. Do not hallucinate or invent information.
5. Quote exact source evidence with document name and page number.
6. If context has conflicting information, mention the conflict.
7. Be concise but complete."""

        user_prompt = f"""CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Answer based ONLY on the context above.
2. Provide specific evidence quotes.
3. If unsure, say "Answer not found in provided documents."
4. Format your response exactly as specified below.

RESPONSE FORMAT:
Answer: [Your answer here]

Evidence:
- Document: [Document Name]
- Page: [Page Number]
- Quote: "[Exact quoted text]"

Confidence: High/Medium/Low
[Brief confidence explanation]"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content
            return self._parse_response(response_text)
            
        except Exception as e:
            return self._call_local_llm(context, question)  # Fallback to local
    
    def _call_local_llm(self, context: str, question: str) -> Dict:
        """Local rule-based answer generator that extracts specific information"""
        
        # Clean and analyze the context
        context_lower = context.lower()
        question_lower = question.lower()
        
        # Extract all document sources from context
        sources = self._extract_sources(context)
        
        # For "what is" questions, look for definitions
        if "what is" in question_lower or "define" in question_lower:
            answer = self._extract_definition(context, question)
            if answer:
                return self._format_response(answer, sources, confidence="High")
        
        # For specific fact questions
        answer = self._extract_specific_answer(context, question)
        if answer:
            return self._format_response(answer, sources, confidence="Medium")
        
        # Default fallback
        return {
            "answer": "I found information in the documents that might answer your question. Please review the retrieved context below.",
            "evidence": [{
                "document": sources[0]["source"] if sources else "Uploaded Document",
                "page": sources[0]["page"] if sources else "1",
                "quote": "See retrieved context for details"
            }],
            "confidence": "Medium",
            "raw_response": context
        }
    
    def _extract_definition(self, context: str, question: str) -> str:
        """Extract definition from context"""
        # Look for patterns like "is defined as", "means", "refers to"
        patterns = [
            r"([^.]*?is[^.]*?definition[^.]*?\.[^.]*)",
            r"([^.]*?defined as[^.]*?\.[^.]*)",
            r"([^.]*?means[^.]*?\.[^.]*)",
            r"([^.]*?refers to[^.]*?\.[^.]*)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # Look for sentences containing the subject
        subject = self._extract_subject(question)
        if subject:
            sentences = context.split('.')
            for sentence in sentences:
                if subject.lower() in sentence.lower() and len(sentence.split()) > 5:
                    return sentence.strip() + "."
        
        return None
    
    def _extract_specific_answer(self, context: str, question: str) -> str:
        """Extract specific answer based on question keywords"""
        keywords = self._extract_keywords(question)
        
        # Split context into sentences
        sentences = re.split(r'[.!?]+', context)
        
        # Score each sentence based on keyword matches
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
            
            score = 0
            sentence_lower = sentence.lower()
            
            for keyword in keywords:
                if keyword in sentence_lower:
                    score += 1
            
            if score > 0:
                scored_sentences.append((score, sentence.strip()))
        
        if scored_sentences:
            # Get the highest scoring sentence
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            return scored_sentences[0][1] + "."
        
        return None
    
    def _extract_subject(self, question: str) -> str:
        """Extract the main subject from question"""
        question_lower = question.lower()
        
        # Remove question words
        question_words = ["what", "is", "are", "does", "do", "can", "could", "will", "would", 
                         "how", "why", "when", "where", "who", "whom", "whose", "which"]
        
        words = question_lower.split()
        subject_words = [word for word in words if word not in question_words and len(word) > 2]
        
        if subject_words:
            return " ".join(subject_words[:3])
        
        return question_lower.strip('?')
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract keywords from question"""
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                     "have", "has", "had", "do", "does", "did", "will", "would", "shall",
                     "should", "may", "might", "must", "can", "could", "of", "in", "on",
                     "at", "by", "for", "with", "about", "against", "between", "into",
                     "through", "during", "before", "after", "above", "below", "to", "from",
                     "up", "down", "in", "out", "on", "off", "over", "under", "again",
                     "further", "then", "once", "here", "there", "when", "where", "why",
                     "how", "all", "any", "both", "each", "few", "more", "most", "other",
                     "some", "such", "no", "nor", "not", "only", "own", "same", "so",
                     "than", "too", "very", "s", "t", "can", "will", "just", "don"}
        
        words = question.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _extract_sources(self, context: str) -> List[Dict]:
        """Extract source information from context"""
        sources = []
        pattern = r'\[Source: ([^,]+), Page: ([^\],]+)'
        
        matches = re.findall(pattern, context)
        for match in matches:
            source = match[0].strip()
            page = match[1].strip()
            sources.append({"source": source, "page": page})
        
        return sources
    
    def _format_response(self, answer: str, sources: List[Dict], confidence: str) -> Dict:
        """Format the response in structured format"""
        evidence = []
        
        if sources:
            for source in sources[:1]:  # Use first source for evidence
                evidence.append({
                    "document": source["source"],
                    "page": source["page"],
                    "quote": answer[:200] + "..." if len(answer) > 200 else answer
                })
        else:
            evidence.append({
                "document": "Provided Context",
                "page": "N/A",
                "quote": answer[:200] + "..." if len(answer) > 200 else answer
            })
        
        return {
            "answer": answer,
            "evidence": evidence,
            "confidence": confidence,
            "raw_response": answer
        }
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response into structured format"""
        parsed = {
            "answer": "",
            "evidence": [],
            "confidence": "Medium",
            "raw_response": response
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Answer:"):
                parsed["answer"] = line.replace("Answer:", "").strip()
                current_section = "answer"
            
            elif line.startswith("Evidence:"):
                current_section = "evidence"
            
            elif line.startswith("- Document:"):
                if current_section == "evidence":
                    evidence = {
                        "document": line.replace("- Document:", "").strip(),
                        "page": "",
                        "quote": ""
                    }
                    parsed["evidence"].append(evidence)
            
            elif line.startswith("- Page:") and parsed["evidence"]:
                parsed["evidence"][-1]["page"] = line.replace("- Page:", "").strip()
            
            elif line.startswith("- Quote:") and parsed["evidence"]:
                quote = line.replace("- Quote:", "").strip()
                if quote.startswith('"') and quote.endswith('"'):
                    quote = quote[1:-1]
                parsed["evidence"][-1]["quote"] = quote
            
            elif line.startswith("Confidence:"):
                confidence = line.replace("Confidence:", "").strip()
                if confidence in ["High", "Medium", "Low"]:
                    parsed["confidence"] = confidence
        
        # If no evidence was parsed but answer was found, add default
        if parsed["answer"] and parsed["answer"] != "Answer not found in provided documents." and not parsed["evidence"]:
            parsed["evidence"] = [{
                "document": "Provided Context",
                "page": "N/A",
                "quote": parsed["answer"][:200] + "..." if len(parsed["answer"]) > 200 else parsed["answer"]
            }]
        
        return parsed