const API_URL = "http://127.0.0.1:8000/predict"; // change this to your deployed API URL

async function checkSentiment(){
  const text = document.getElementById("reviewInput").value.trim();
  const btn = document.getElementById("checkBtn");
  const resultBox = document.getElementById("result");
  const errorBox = document.getElementById("error");

  errorBox.style.display = "none";
  resultBox.style.display = "none";

  if(!text){
    errorBox.textContent = "Please write a review first.";
    errorBox.style.display = "block";
    return;
  }

  btn.disabled = true;
  btn.textContent = "Checking...";

  try{
    const res = await fetch(API_URL, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text})
    });

    if(!res.ok){
      const err = await res.json();
      throw new Error(err.detail || "Something went wrong.");
    }

    const data = await res.json();
    const isPositive = data.sentiment === "positive";
    const verdictBox = document.getElementById("verdictBox");

    verdictBox.className = "verdict " + (isPositive ? "positive" : "negative");
    document.getElementById("verdictIcon").textContent = isPositive ? "🍿" : "🎭";
    document.getElementById("verdictValue").textContent = isPositive ? "Positive" : "Negative";
    document.getElementById("confidenceFill").style.width = (data.confidence * 100) + "%";
    document.getElementById("confidenceLabel").textContent = `Confidence: ${(data.confidence*100).toFixed(1)}%`;

    resultBox.style.display = "block";
  }catch(e){
    errorBox.textContent = e.message.includes("fetch") ? "Could not reach the API — is it running?" : e.message;
    errorBox.style.display = "block";
  }finally{
    btn.disabled = false;
    btn.textContent = "Check Sentiment";
  }
}
