from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer, util
import json

app = FastAPI()

# Load model and data once
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lighter model for Render
with open("freelancer_profiles.json", "r") as f:
    freelancer_data = json.load(f)

@app.post("/recommend")
async def recommend_freelancers(request: Request):
    data = await request.json()
    job_description = data.get("description")

    freelancer_texts = [
        f"{freelancer['Skill']}. {freelancer['Bio']}. Rating: {freelancer['Rating']} stars."
        for freelancer in freelancer_data
    ]

    job_embedding = model.encode(job_description, convert_to_tensor=True)
    freelancer_embeddings = model.encode(freelancer_texts, convert_to_tensor=True)

    cosine_scores = util.cos_sim(job_embedding, freelancer_embeddings)
    scores = cosine_scores[0].cpu().numpy()
    sorted_indices = scores.argsort()[::-1]

    top_freelancers = []
    for idx in sorted_indices[:10]:
        freelancer = freelancer_data[idx]
        top_freelancers.append({
            "Name": freelancer["Name"],
            "Skill": freelancer["Skill"],
            "Location": freelancer["Location"],
            "Rating": freelancer["Rating"],
            "Bio": freelancer["Bio"],
            "Similarity": round(scores[idx], 4)
        })

    return {"results": top_freelancers}