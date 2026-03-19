# Hack4IMPACTTrack2-hehe-we-code


### Team Details
* **Team Name:** hehe we code
* **Members:** Aditya Tiwari, Tanush Rawat, Nikhil Sharma, Manik Pandey
* **Domain:** Smart Agriculture & Food Security

### Approved Problem Statement
**Vani-Check: Offline Multimodal AI for Objective Quality Grading and Fair-Price Discovery in the Vegetable Supply Chain.**

Small-scale vendors in India lose up to 40% of income due to subjective grading and price exploitation. Operating in "data-dark" regions with zero internet, they lack tools to certify produce quality. Vani-Check provides an offline-first AI auditor using Computer Vision (YOLOv11) and Local LLMs (Llama-3) to provide objective grading and smart pricing based on cached Agmarknet data.

### Tech Stack
* **Vision:** YOLOv11 (Optimized for Radeon 780M GPU)
* **LLM:** Llama-3-8B (4-bit GGUF via llama.cpp)
* **Speech:** Vosk Offline STT
* **Hardware Acceleration:** AMD Ryzen 7 8000 Series (XDNA NPU)
* **Backend:** FastAPI with Local SQLite Cache
