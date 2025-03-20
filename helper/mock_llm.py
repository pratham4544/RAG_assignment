from typing import List, Dict
import time
import random

class MockLLM:
    def __init__(self):
        self.request_count = 0
        self.rate_limit = 10  # requests per minute
        self.last_request_time = 0
        
    def _check_rate_limit(self):
        """
        Check if rate limit is exceeded.
        
        Raises:
            Exception: If rate limit is exceeded
        """
        current_time = time.time()
        time_diff = current_time - self.last_request_time
        
        # If less than a minute has passed since the first request in this minute
        if time_diff < 60 and self.request_count >= self.rate_limit:
            sleep_time = 60 - time_diff
            raise Exception(f"Rate limit exceeded. Try again in {sleep_time:.2f} seconds")
            
        # Reset counter if a minute has passed
        if time_diff >= 60:
            self.request_count = 0
            
        self.last_request_time = current_time
        self.request_count += 1
        
    def _simulate_error(self):
        """
        Simulate random errors that might occur in an LLM API.
        
        Returns:
            tuple: (bool, str) - (error occurred, error message)
        """
        # Simulate different error scenarios with low probability
        error_chance = random.random()
        
        if error_chance < 0.05:  # 5% chance of timeout
            return True, "Request timed out"
        elif error_chance < 0.08:  # 3% chance of server error
            return True, "Server error: Internal processing failed"
        elif error_chance < 0.10:  # 2% chance of invalid input
            return True, "Invalid input: Input too long or contains invalid characters"
            
        return False, ""
        
    def generate_response(self, prompt: str, context: List[str]) -> dict:
        """
        Simulate LLM response generation.
        Include rate limiting and error simulation.
        
        Args:
            prompt (str): User's question
            context (List[str]): Retrieved context chunks
            
        Returns:
            dict: Response data containing generated text and metadata
        """
        # Check rate limit
        try:
            self._check_rate_limit()
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
            
        # Simulate processing time
        processing_time = random.uniform(0.5, 2.0)  # Between 0.5 and 2 seconds
        time.sleep(processing_time)
        
        # Simulate potential errors
        has_error, error_message = self._simulate_error()
        if has_error:
            return {
                "success": False,
                "error": error_message,
                "data": None
            }
            
        # Generate a mock response based on the context and prompt
        response_text = f"Based on the provided context, "
        
        # Add some content from the context
        if context:
            # Get some snippets from context
            snippets = []
            for i, chunk in enumerate(context):
                if len(chunk) > 50:  # Make sure the chunk has some content
                    # Take a short excerpt from the chunk
                    excerpt = chunk[:100] + "..."
                    snippets.append(excerpt)
                    
                    # Limit the number of snippets
                    if i >= 2:
                        break
                        
            if snippets:
                response_text += "I found the following relevant information: "
                response_text += " ".join(snippets)
            else:
                response_text += "I couldn't find specific details related to your query."
        else:
            response_text += "I don't have enough context to answer your question adequately."
            
        # Add some reflection on the prompt
        response_text += f" Regarding your question about '{prompt}', "
        response_text += "this appears to be related to the document's content. "
        response_text += "Please note that this is a simulated response for demonstration purposes."
        
        return {
            "success": True,
            "error": None,
            "data": {
                "text": response_text,
                "processing_time": processing_time,
                "tokens_used": len(response_text.split())
            }
        } 