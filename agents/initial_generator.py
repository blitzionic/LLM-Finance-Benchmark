from base_agent import Agent

class InitialGeneratorAgent(Agent):
    def system_prompt(self):
        return "You are a knowledgeable finance expert. Provide a comprehensive answer to the following finance question."

    def process(self, question):
        prompt = f"Please provide a detailed solution to the following question:\n\n{question}"
        initial_solution = self.generate_response(prompt)
        return initial_solution
