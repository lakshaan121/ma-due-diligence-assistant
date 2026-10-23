# Interview Preparation

Short, defensible answers tied to what was actually built. Practice
saying these out loud, not just reading them.

**1. What problem does your project solve?**
It speeds up the first-pass document review in M&A due diligence by
letting a user ask natural-language questions over uploaded company
documents and get answers grounded in, and cited to, the actual text.

**2. Why did you choose M&A due diligence?**
It's a realistic, document-heavy domain where keyword search genuinely
falls short, and it lets me demonstrate RAG plus simple domain-specific
analysis (ratios, risk flags) in one coherent project.

**3. What is RAG?**
Retrieval-Augmented Generation: retrieve relevant text, augment the
LLM's prompt with it, then generate an answer grounded in that text
instead of the model's training data alone.

**4. Why did you use RAG?**
The LLM has never seen my uploaded PDFs. RAG lets it answer about them
anyway, by handing it the relevant chunks at question time, and keeps
answers traceable to a real source.

**5. What happens when a PDF is uploaded?**
`pdf_processor.py` extracts text page-by-page using pypdf, tagging
each page with the document name and page number.

**6. What is text chunking?**
Splitting long extracted text into smaller overlapping pieces (~1000
characters, 150 overlap) so each piece is small enough to embed
precisely and fits comfortably in the LLM's prompt.

**7. Why do you split the document?**
A single embedding vector per page would blur together every topic on
that page, making retrieval vague. Smaller chunks give each vector a
narrower, more accurate meaning.

**8. What are embeddings?**
Numerical vectors that represent the meaning of text. Semantically
similar text (even with different words) produces vectors that are
close together.

**9. Why do you need embeddings?**
They're what make semantic (meaning-based) search possible, instead of
being limited to exact keyword matches.

**10. What is Sentence-Transformers?**
A library providing pre-trained models specifically for producing good
sentence/paragraph-level embeddings.

**11. Why did you use all-MiniLM-L6-v2?**
It's small (~80MB), runs fast on CPU with no GPU needed, and is
accurate enough for this scope — appropriate for a student project
without paid infrastructure.

**12. What is FAISS?**
A library from Meta for fast similarity search over large sets of
vectors.

**13. Why use FAISS?**
It's the standard tool for this exact job — efficient nearest-neighbor
search — and well documented, which matters for a project I need to
explain clearly.

**14. How does similarity search work?**
The question is embedded into the same 384-dimensional space as the
chunks. FAISS finds the chunk vectors with the highest similarity
(smallest angular distance) to the question vector.

**15. What is cosine similarity?**
A measure of the angle between two vectors, ignoring their length —
it captures whether two pieces of text point in the same semantic
direction. Since my vectors are normalized to unit length, a simple
inner product (what `IndexFlatIP` computes) equals cosine similarity.

**16. What is an LLM?**
A large language model — a neural network trained on large amounts of
text that can generate coherent, context-aware natural language.

**17. What does the LLM do in your project?**
It takes the retrieved chunks plus the user's question and writes a
natural-language answer, instructed to use only that context.

**18. Does the LLM read the entire PDF?**
No — only the top-k chunks FAISS retrieved for that specific question,
usually a few hundred to a couple thousand characters total.

**19. How does your system reduce hallucination?**
The system prompt explicitly restricts the LLM to the supplied context
and gives it an exact fallback sentence — "Insufficient information in
the provided documents" — to use when the context doesn't answer the
question.

**20. How do you show the source of an answer?**
Every chunk carries its original document name and page number from
the extraction step onward. These are displayed alongside the answer
as expandable source cards.

**21. What happens if the answer isn't present?**
The LLM is instructed to say so explicitly rather than guessing —
that's the hallucination-control fallback from question 19.

**22. What financial calculations does your system perform?**
Net Profit Margin, Debt-to-Asset Ratio, and Debt-to-Equity Ratio, all
from figures the user enters manually — never fabricated if an input
is missing.

**23. What are the limitations of your project?**
No OCR for scanned PDFs, no automatic extraction of financial figures,
risk flagging is keyword-based (not predictive), question
classification is rule-based (not a trained model), and it's only been
tested on sample documents, not real company filings at scale.

**24. What would you improve in the future?**
OCR support, automatic structured financial data extraction, and
PDF export of the report — all listed as explicit Future Enhancements
rather than overstated as done.

**25. What part of the project did you personally implement?**
*(Fill this in honestly based on what you actually understood and
typed/debugged yourself — this is the one question you must answer
in your own words, not from this document.)*
