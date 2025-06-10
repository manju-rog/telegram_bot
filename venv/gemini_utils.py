import requests
import json

# Replace with your actual Gemini API key
GEMINI_API_KEY = "AIzaSyB7G-ePwKXFwzl1wswUT3eTYlU9-Kzr3mQ"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

# Median values (these can be updated with actual values from your dataset)
MEDIAN_VALUES = {
    'sqft': 1500,  # Median sqft value
    'bedrooms': 3,  # Median bedroom value
    'location': 'Unknown Location',  # Default location if missing
    'MedInc': 4.5,  # Median Income in block group
    'HouseAge': 30,  # Median House Age in block group (years)
    'AveRooms': 6,  # Median Rooms per household
    'AveBedrms': 3,  # Median Bedrooms per household
    'Population': 5000,  # Median Population per block group
    'AveOccup': 2.5,  # Median Average Occupants
    'Latitude': 37.7749,  # Example Latitude for California
    'Longitude': -122.4194  # Example Longitude for California
}

def extract_features_from_text(prompt):
    """Extract house features like bedrooms, location, size from user input using Gemini API."""
    body = {
        "contents": [{
            "parts": [{
                "text": f"""
                Extract the following structured details from the user's house description:
                - bedrooms
                - location
                - size (in sqft)
                Return a JSON like:
                {{
                  "bedrooms": 2,
                  "location": "Bangalore",
                  "sqft": 1200
                }}

                User's input: {prompt}
                """
            }]
        }]
    }

    try:
        # Sending the request to Gemini API
        response = requests.post(GEMINI_URL, json=body)
        response.raise_for_status()  # Will raise an error for bad status codes
        content = response.json()

        # Debugging: print the raw response
        print("Raw Gemini Response:", content)

        # Extract the content from the response
        text = content['candidates'][0]['content']['parts'][0]['text']

        # Clean the text (strip unwanted parts)
        clean_text = text.strip("```json\n").strip("```").strip()  # Remove markdown parts

        # Try parsing the cleaned text as JSON
        try:
            extracted = json.loads(clean_text)

            # Handle missing values gracefully by using the median values
            extracted['location'] = extracted.get('location', MEDIAN_VALUES['location'])  # Use median if missing
            extracted['sqft'] = extracted.get('sqft', MEDIAN_VALUES['sqft'])  # Use median sqft if missing
            extracted['bedrooms'] = extracted.get('bedrooms', MEDIAN_VALUES['bedrooms'])  # Use median bedrooms if missing

            # Debugging: print the extracted features
            print("Extracted Features:", extracted)

            return extracted
        except json.JSONDecodeError as json_error:
            print(f"Error parsing response JSON: {json_error}")
            return None

    except Exception as e:
        print(f"Error processing with Gemini API: {e}")
        return None