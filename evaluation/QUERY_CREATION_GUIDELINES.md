# Test Query Creation Guidelines

This document provides detailed guidance on creating effective test queries for evaluating large language models (LLMs). Well-designed test queries are essential for thorough and meaningful model evaluation.

## Query Structure

Each test query should be formatted as a JSON object with the following fields:

```json
{
  "query_id": "q11",
  "query": "What's the best way to avoid jet lag when traveling across multiple time zones?",
  "category": "travel_health",
  "difficulty": "medium"
}
```

### Field Descriptions

- **query_id**: A unique identifier for the query (e.g., "q11", "destination_1", "safety_3")
- **query**: The actual text of the question to be sent to the models
- **category**: A descriptive category that groups similar types of queries
- **difficulty**: An assessment of the query's complexity ("easy", "medium", or "hard")

## Creating Balanced Test Sets

An effective test set should:

1. **Cover diverse topics** within your domain
2. **Include varying levels of difficulty**
3. **Test different aspects** of model capabilities
4. **Represent realistic user queries**

## Guidelines for Travel Assistant Queries

Since our evaluation framework is designed for a travel assistant, here are specific guidelines for creating travel-related test queries:

### 1. Domain Coverage

Include queries across these key travel categories:

| Category | Description | Example |
|----------|-------------|---------|
| Destination Recommendations | Suggestions for places to visit | "What are the must-visit places in Tokyo for a first-time visitor?" |
| Itinerary Planning | Creating travel schedules | "Can you suggest an itinerary for 3 days in Barcelona?" |
| Food Recommendations | Local cuisine and dining | "What are some local dishes I should try in Thailand?" |
| Family Travel | Travel with children or families | "I'm traveling with young children to Costa Rica. What activities would be appropriate and safe?" |
| Travel Planning | Logistics, timing, preparations | "What's the best time of year to visit Machu Picchu and what should I know about altitude sickness?" |
| Transportation | Getting around at destinations | "Can you explain the public transportation system in London? How do I get from Heathrow to central London?" |
| Cultural Experiences | Authentic local experiences | "I want to experience authentic local culture in Marrakech. What hidden gems should I visit?" |
| Safety | Travel safety and precautions | "What safety precautions should I take when traveling to Rio de Janeiro?" |
| Sustainable Travel | Eco-friendly travel options | "I'm looking for sustainable and eco-friendly travel options in New Zealand. What would you recommend?" |
| Cultural Awareness | Cultural norms and etiquette | "What are common cultural faux pas to avoid in Japan?" |
| Budget Travel | Cost-effective travel | "How can I experience Paris on a tight budget?" |
| Travel Health | Health concerns and preparations | "What vaccinations do I need for a trip to Tanzania?" |

### 2. Difficulty Levels

When assigning difficulty levels, consider these factors:

#### Easy (Straightforward queries)
- Simple, direct questions
- Common knowledge in the travel domain
- Single-focus questions
- Example: "What are popular beaches in Bali?"

#### Medium (Moderately complex)
- Questions requiring some specific knowledge
- Multiple related components
- Some nuance or context needed
- Example: "What are the best areas to stay in Barcelona for a family with teenagers who want to be close to cultural attractions but avoid noisy areas?"

#### Hard (Complex or challenging)
- Questions requiring specialized knowledge
- Multiple constraints or complex planning
- Nuanced trade-offs or considerations
- Example: "I'm planning a 10-day trip to Vietnam during rainy season with a limited budget, focusing on cultural experiences and photography opportunities. How should I structure my itinerary to maximize experiences while minimizing weather disruptions?"

### 3. Query Types to Include

For a comprehensive evaluation, include queries that test:

#### Factual Knowledge
- Historical information about destinations
- Geographic details
- Cultural facts
- Example: "What's the historical significance of Angkor Wat in Cambodia?"

#### Planning & Organization
- Itinerary creation
- Scheduling activities
- Logistical planning
- Example: "How should I plan a 5-day trip to Rome to see all the major historical sites?"

#### Reasoning & Recommendations
- Personalized suggestions
- Balancing trade-offs
- Contextual advice
- Example: "I'm traveling to Thailand for 2 weeks in July with a 5-year-old. Where should we go to balance interesting experiences for me with kid-friendly activities, while avoiding the worst of the monsoon season?"

#### Multi-part Questions
- Questions with multiple distinct components
- Example: "What's the best time to visit Morocco, what clothing should I pack, and how should I handle currency exchange?"

#### Open-ended Exploration
- Questions that invite creative or diverse responses
- Example: "What are some unique or overlooked destinations in Eastern Europe for someone interested in history and architecture?"

## Best Practices for Query Creation

### Do:
- ✅ Use clear, natural language
- ✅ Include some specific constraints or context
- ✅ Vary query length and complexity
- ✅ Include regional diversity (different countries/continents)
- ✅ Include seasonal considerations (different times of year)
- ✅ Test for cultural sensitivity and awareness
- ✅ Include queries relevant to different traveler types (solo, family, luxury, budget)

### Don't:
- ❌ Use overly technical jargon
- ❌ Create queries with potentially harmful outputs
- ❌ Include leading questions that suggest a particular answer
- ❌ Make queries so vague that success criteria are unclear
- ❌ Create queries that are so specific that only one correct answer exists
- ❌ Include personally identifiable information

## Examples of Well-Crafted Queries

### Easy:
```json
{
  "query_id": "food_1",
  "query": "What are some must-try street foods in Bangkok?",
  "category": "food_recommendations",
  "difficulty": "easy"
}
```

### Medium:
```json
{
  "query_id": "itinerary_3",
  "query": "I have a 12-hour layover in Istanbul. What's a reasonable itinerary to see the main highlights without missing my connecting flight?",
  "category": "itinerary_planning",
  "difficulty": "medium"
}
```

### Hard:
```json
{
  "query_id": "planning_5",
  "query": "I'm planning a 2-week trip to Japan in April with my elderly parents and two teenagers. We want to see cherry blossoms, experience both traditional and modern Japan, and accommodate my father's limited mobility. How should we structure our trip?",
  "category": "travel_planning",
  "difficulty": "hard"
}
```

## Extending Your Test Set

When creating additional queries to extend the default test set:

1. Save them in the same JSON format in your own file
2. Ensure each query has a unique `query_id`
3. Try to maintain a balance of categories and difficulty levels
4. Test your queries to make sure they're clear and unambiguous

You can add your custom queries to the evaluation by:

```bash
# Copy your custom queries to the evaluation data directory
cp your_custom_queries.json L200-upskilling/evaluation/data/

# Edit the evaluation script to use your custom queries
# In run_evaluation.py, update the path to your query file
```

## Using Test Queries for Targeted Evaluation

Different sets of test queries can help evaluate specific aspects of model performance:

- **General capability assessment**: Broad coverage across categories
- **Specialized testing**: Deep focus on one category (e.g., 10 different safety-related queries)
- **Comparison testing**: Same queries across different models or model versions
- **Fine-tuning evaluation**: Before/after testing to measure improvements

By following these guidelines, you'll create test queries that provide meaningful insights into model performance and help identify areas for improvement. 