import json
from transformers import pipeline

# Function to load keywords from a file
def load_keywords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        keywords = [line.strip().lower() for line in f.readlines()]
    return keywords

# Function to load the JSON file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Function to save the JSON file
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Define paths for keyword files
keyword_files = {
    "Greetings": "keywords/greetings.txt",
    "Need Analysis": "keywords/need_analysis.txt",
    "Payment Discussion": "keywords/payment_discussion.txt",
    "Form Filling Discussion": "keywords/form_filling_discussion.txt",
    "Upsell Done": "keywords/upsell_done.txt",
    "MLI Differentiators": "keywords/mli_differentiators.txt"
}

# Function to extract summary from the transcription
def extract_summary(transcription):
    summary = []
    for term, file_path in keyword_files.items():
        keywords = load_keywords(file_path)
        val = 0
        for sentence in transcription['sentList']:
            if any(keyword in sentence['sentence'].lower() for keyword in keywords):
                val = 1
                break
        summary.append({"name": term, "tag": file_path.split('/')[-1].split('.')[0].upper(), "val": val, "wt": 1})
    return summary

# Function to perform sentiment analysis
def analyze_sentiment(transcription):
    classifier = pipeline('sentiment-analysis')
    full_text = " ".join([sentence['sentence'] for sentence in transcription['sentList']])
    result = classifier(full_text)
    return result[0]['label']

# Main function
def main(input_file, output_file):
    # Load the transcription file
    transcription_data = load_json(input_file)

    # Extract summary
    call_summary = extract_summary(transcription_data)

    # Analyze sentiment
    sentiment = analyze_sentiment(transcription_data)

    # Generate the output JSON
    output_data = {
        "transcription": transcription_data,
        "callSummary": call_summary,
        "statusCode": 200,
        "statusMsg": "success",
        "sentiment": sentiment,
        "summaryTotal": len(call_summary),
        "summaryMax": len(call_summary)
    }

    # Save the output JSON file
    save_json(output_data, output_file)
    print(f"Output saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_file = "transcription.json"  # Path to your input JSON file
    output_file = "output_summary.json"  # Path to your output JSON file
    main(input_file, output_file)
