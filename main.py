from fastapi import FastAPI, HTTPException, Cookie, Request, Response,Depends
from pydantic import BaseModel
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Optional
from uuid import uuid4

# FastAPI application instance
app = FastAPI()


# Define request/response models
class QueryRequest(BaseModel):
    user_input: str


class Source(BaseModel):
    url: str
    paragraph: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    citations: str  # For in-text citations
    session_id: str  # New field for the session_id



# Setup LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Load FAISS index (Vectorstore)
embeddings = OpenAIEmbeddings()
faiss_vectorstore = FAISS.load_local("faiss_index",
                                     embeddings,
                                     allow_dangerous_deserialization=True)
retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

# Contextualize question prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, 
formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt)

# Updated QA System Prompt for OSHA-style responses with citations and chat memory
qa_system_prompt = """
You are an expert in OSHA regulations and workplace safety, specializing in providing clear, accurate, and actionable guidance based on OSHA standards, regulations, and letters of interpretation. 
You assist professionals with compliance, incident reporting, safety audits, and training programs. Your tone is professional yet approachable, making complex safety topics easy to understand and implement.

Consider the full context of the chat history to provide a comprehensive response. If relevant, incorporate information from previous interactions in the chat, not just the current question. Formulate the response based on all relevant context, ensuring continuity in the conversation.

Based on the provided context and chat history, answer the question thoroughly and include citations that MUST be from the given sources in this format:
<answer_text> [1][2]...

Provide a numbered source list at the end of your response with the following format:
Source List:
1 -<URL1>
2 -<URL2>

Here is the context and chat history: {context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Setup the question-answer chain with citations
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create the retrieval chain
rag_chain = create_retrieval_chain(history_aware_retriever,
                                   question_answer_chain)

# Manage chat history
store = {}  # This will store session histories using session IDs as keys


def get_or_create_session_id(session_id: Optional[str] = Cookie(None)) -> str:
    """Retrieve or create a new session ID if not provided."""
    if not session_id:
        session_id = str(
            uuid4())  # Generate a unique session ID if not present
    return session_id


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory(
        )  # Initialize chat history for this session
    print(f"History for {session_id}: {store[session_id].messages}")
    return store[session_id]


# Create the conversational RAG chain with history and citations
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


def assign_citations_to_segments(answer: str,
                                 source_documents: list) -> (str, list):
    """
    Assigns in-text citations to segments of the answer based on the context of each source document.
    Returns the annotated answer and a list of source URLs only.
    """
    segments = answer.split(
        ". ")  # Split the answer by sentences for simplicity
    annotated_answer = []
    source_list = []
    sources_metadata = []
    citation_index = 1

    for segment in segments:
        best_match = None
        highest_overlap = 0

        # Find the source that has the highest overlap with the segment
        for doc in source_documents:
            content = doc.page_content.lower()
            url = doc.metadata.get('url', 'N/A')

            # Calculate the overlap score based on shared words
            overlap = len(
                set(segment.lower().split()).intersection(content.split()))
            if overlap > highest_overlap:
                highest_overlap = overlap
                best_match = url

        # Add the citation to the segment and record the source URL
        if best_match:
            annotated_segment = f"{segment} [{citation_index}]"
            source_list.append(f"{citation_index} - {best_match}")

            # Only add URL to sources metadata
            sources_metadata.append({"url": best_match})
            citation_index += 1
        else:
            annotated_segment = segment

        annotated_answer.append(annotated_segment)

    # Format the answer text
    formatted_answer = ". ".join(annotated_answer)

    # Format the source list
    formatted_source_list = "Source List:\n" + "\n".join(source_list)

    # Combine the answer and source list
    final_output = f"{formatted_answer}\n\n{formatted_source_list}"

    # Ensure we return two values
    return final_output, sources_metadata


# FastAPI POST endpoint to ask questions
@app.post("/ask", response_model=QueryResponse)
async def ask_question(
    query: QueryRequest, 
    response: Response, 
    session_id: Optional[str] = Cookie(None)
):
    try:
        # Generate or retrieve session ID
        if not session_id:
            session_id = str(uuid4())  # Create a new session ID if none is provided
            response.set_cookie(key="session_id", value=session_id)  # Set cookie with session ID

        print(f"Session ID for request: {session_id}")  # Debugging: Check if session ID is consistent

        # Get the current chat history for this session
        chat_history = get_session_history(session_id)
        print(f"History for {session_id}: {chat_history.messages}")  # Debugging: Check if history is stored

        # Add the user's input to the chat history (before calling the model)
        chat_history.add_user_message(query.user_input)

        # Generate the answer using the conversational RAG chain
        result = conversational_rag_chain.invoke(
            {"input": query.user_input},  # Pass the user input
            config={"configurable": {"session_id": session_id}}  # Ensure session ID is passed
        )

        # Extract the answer and source documents
        answer = result.get("answer", "No answer generated")
        source_documents = result.get("source_documents", [])

        # Add the model's answer to the chat history (after generating the response)
        chat_history.add_ai_message(answer)

        # Prepare the response with citations
        annotated_answer, sources_metadata = assign_citations_to_segments(answer, source_documents)

        # Return the response with session_id as a separate field
        return QueryResponse(
            answer=annotated_answer,
            sources=sources_metadata,
            citations=annotated_answer,
            session_id=session_id  # Now returning session_id as a separate field
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# FastAPI root endpoint for session management
@app.get("/")
async def root(response: Response, session_id: Optional[str] = Cookie(None)):
    if not session_id:
        session_id = str(uuid4())  # Generate a new session ID if not provided
        response.set_cookie(key="session_id", value=session_id)  # Set the cookie with session ID
    return {
        "message": "Welcome to the RAG-based Chat API with Citations",
        "session_id": session_id
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
