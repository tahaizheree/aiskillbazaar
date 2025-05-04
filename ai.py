import json
from sentence_transformers import SentenceTransformer, util

# Load the RoBERTa model
model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

# Load freelancer data from file
with open("freelancer_profiles.json", "r") as f:
    freelancer_data = json.load(f)

# Define the job description
job_description = "I want someone to optimize my website ranking in google"

# Create a combined profile text for each freelancer: Skill + Bio + Rating
freelancer_texts = []
for freelancer in freelancer_data:
    skill = freelancer.get("Skill", "")
    bio = freelancer.get("Bio", "")
    rating = freelancer.get("Rating", 0)
    combined_text = f"{skill}. {bio}. Rating: {rating} stars."
    freelancer_texts.append(combined_text)

# Encode job description and freelancer texts
job_embedding = model.encode(job_description, convert_to_tensor=True)
freelancer_embeddings = model.encode(freelancer_texts, convert_to_tensor=True)

# Compute cosine similarity
cosine_scores = util.cos_sim(job_embedding, freelancer_embeddings)
scores = cosine_scores[0].cpu().numpy()
sorted_indices = scores.argsort()[::-1]

# Retrieve top N matches
top_n = 10
top_freelancers = []
for idx in sorted_indices[:top_n]:
    freelancer = freelancer_data[idx]
    similarity_score = round(scores[idx], 4)
    top_freelancers.append({
        "Name": freelancer["Name"],
        "Skill": freelancer["Skill"],
        "Location": freelancer["Location"],
        "Rating": freelancer["Rating"],
        "Bio": freelancer["Bio"],
        "Similarity": similarity_score
    })

# Print the top recommendations
for freelancer in top_freelancers:
    print(f"{freelancer['Name']} ({freelancer['Skill']}) - Similarity: {freelancer['Similarity']}")
    print(f"Rating: {freelancer['Rating']} stars")
    print(f"Bio: {freelancer['Bio']}\n")