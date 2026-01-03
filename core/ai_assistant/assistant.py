    def suggest_next(self):
        """Suggests next logical scientific question or analysis step."""
        prompt = "Based on current context, suggest the next analytical step in optics experiment."
        response = self.pipe(prompt, max_new_tokens=100, temperature=0.5)[0]['generated_text']
        return response
