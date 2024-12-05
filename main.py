import json
import os
from datetime import datetime
from pathlib import Path

import openai
from langchain.chains import ConversationalRetrievalChain
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI as LangchainOpenAI
from langchain_openai import OpenAIEmbeddings

# Configuration
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
openai.api_key = OPENAI_API_KEY


class PDFChatbot:
    def __init__(self, storage_directory: str, bot_name: str = "EduBot"):
        """
        Initialize the chatbot with fallback mechanism

        Args:
            storage_directory (str): Directory containing processed vectorstore
            bot_name (str): Name of the chatbot
        """
        self.storage_directory = Path(storage_directory)
        self.vectorstore_path = self.storage_directory / "vectorstore"
        self.chat_history_path = self.storage_directory / "chat_history"

        # Ensure directories exist
        self.chat_history_path.mkdir(parents=True, exist_ok=True)

        self.bot_name = bot_name
        self.vectorstore = None
        self.embeddings = OpenAIEmbeddings()

        # Chat session tracking
        self.current_session_id = None

        # Try to load vectorstore
        try:
            if self.vectorstore_path.exists():
                self.vectorstore = FAISS.load_local(
                    str(self.vectorstore_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
        except Exception as e:
            print(f"Warning: Could not load vectorstore: {str(e)}")

        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Initialize the conversation chain (if vectorstore exists)
        if self.vectorstore:
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=LangchainOpenAI(temperature=0.7),
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory,
                return_source_documents=True,
                verbose=False
            )
        else:
            self.chain = None

    def start_session(self):
        """Start a new chat session"""
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.memory.clear()
        return self.current_session_id

    # def get_response(self, user_input: str):
    #     """
    #     Get response with fallback to direct OpenAI API
    #
    #     Args:
    #         user_input (str): User's query
    #
    #     Returns:
    #         Dict containing answer and optional sources
    #     """
    #     # First try vector store retrieval if available
    #     if self.chain:
    #         try:
    #             response = self.chain.invoke({"question": user_input})
    #             # Check if sources are relevant
    #             if response.get("source_documents"):
    #                 print(response)
    #                 return response
    #
    #         except Exception as e:
    #             print(f"Vectorstore retrieval failed: {str(e)}")
    #
    #     # Fallback to direct OpenAI API
    #     try:
    #         chat_response = openai.chat.completions.create(
    #             model="gpt-3.5-turbo",
    #             messages=[
    #                 {"role": "system",
    #                  "content": "You are a helpful assistant trained to answer questions based on provided context. If no specific context is available, provide a helpful and informative answer."},
    #                 {"role": "user", "content": user_input}
    #             ]
    #         )
    #
    #         answer = chat_response.choices[0].message.content
    #
    #         return {
    #             "answer": answer,
    #             "source_documents": []  # No sources in direct API call
    #         }
    #
    #     except Exception as e:
    #         return {
    #             "answer": f"Sorry, I couldn't retrieve an answer. Error: {str(e)}",
    #             "source_documents": []
    #         }
    def get_response(self, user_input: str):
        """
        Get response with improved context retrieval and response generation

        Args:
            user_input (str): User's query

        Returns:
            Dict containing answer and optional sources
        """
        # Debug print to check vectorstore status
        print(f"Vectorstore status: {'Exists' if self.vectorstore else 'Not Loaded'}")

        # First try vector store retrieval if available
        if self.chain and self.vectorstore:
            try:
                # Perform similarity search first
                similar_docs = self.vectorstore.similarity_search(user_input, k=3)
                print(f"Similar documents found: {len(similar_docs)}")

                # If similar documents exist, create context
                if similar_docs:
                    # Combine document contents to create context
                    context = "\n\n".join([doc.page_content for doc in similar_docs])

                    # Use OpenAI to generate a response based on context
                    try:
                        context_response = openai.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system",
                                    "content": f"You are a helpful assistant for IvyTech. Use the following context to answer the user's question as accurately as possible. If you cannot find a precise answer, Try your Best and keep it brief.\n\nContext:\n{context}"
                                },
                                {"role": "user", "content": user_input}
                            ],
                            max_tokens=300
                        )

                        answer = context_response.choices[0].message.content

                        return {
                            "answer": answer,
                            "source_documents": similar_docs
                        }

                    except Exception as e:
                        print(f"Context-based response generation failed: {str(e)}")

            except Exception as e:
                print(f"Vectorstore retrieval error: {str(e)}")

        # Fallback to direct OpenAI API
        try:
            print("Chatgpt called")
            chat_response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant trained to answer questions about IvyTech courses and campus. Provide the most helpful information you can."
                    },
                    {"role": "user", "content": user_input}
                ]
            )

            answer = chat_response.choices[0].message.content

            return {
                "answer": answer,
                "source_documents": []
            }

        except Exception as e:
            print(f"OpenAI API call failed: {str(e)}")
            return {
                "answer": f"I'm having trouble retrieving an answer. The query might be outside my current knowledge. Error: {str(e)}",
                "source_documents": []
            }

    # def check_vectorstore_health(self):
    #     """
    #     Check and report on vectorstore health
    #     """
    #     print("Vectorstore Health Check:")
    #     print(f"Vectorstore Path: {self.vectorstore_path}")
    #     print(f"Vectorstore Exists: {self.vectorstore_path.exists()}")
    #
    #     if self.vectorstore:
    #         try:
    #             # Check number of documents
    #             total_docs = self.vectorstore.index.ntotal
    #             print(f"Total Documents in Vectorstore: {total_docs}")
    #
    #             # Perform diagnostic search
    #             test_query = "IvyTech campus"
    #             similar_docs = self.vectorstore.similarity_search(test_query, k=3)
    #
    #             print(f"Diagnostic search results for '{test_query}':")
    #             print(f"Number of similar documents found: {len(similar_docs)}")
    #
    #             # Print out document metadata
    #             for doc in similar_docs:
    #                 print("\nDocument Metadata:")
    #                 print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    #                 print(f"Content Preview: {doc.page_content[:200]}...")
    #
    #         except Exception as e:
    #             print(f"Vectorstore health check failed: {str(e)}")
    #     else:
    #         print("Vectorstore is not initialized!")

    def save_chat_history(self, session_id: str = None):
        """Save chat history to file"""
        session_id = session_id or self.current_session_id
        if not session_id:
            return

        history_file = self.chat_history_path / f"chat_history_{session_id}.json"

        history = []
        for message in self.memory.chat_memory.messages:
            history.append({
                "role": "user" if message.type == "human" else "assistant",
                "content": message.content,
                "timestamp": datetime.now().isoformat()
            })

        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

    def clear_history(self):
        """Clear current session history"""
        self.memory.clear()
        if self.current_session_id:
            history_file = self.chat_history_path / f"chat_history_{self.current_session_id}.json"
            if history_file.exists():
                os.remove(history_file)


# Flask Application
app = Flask(__name__)
CORS(app)

# Initialize the chatbot
chatbot = PDFChatbot(
    storage_directory="./storage_directory",
    bot_name="IvyTech Chat Bot"
)


@app.route('/')
def index():
    """Render the main chat interface"""
    # Start a new session when webpage loads
    session_id = chatbot.start_session()
    return render_template('chat.html', session_id=session_id)


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        user_message = data.get('message', '')

        # Ensure a session is active
        if not chatbot.current_session_id:
            chatbot.start_session()

        # Get response from chatbot
        response = chatbot.get_response(user_message)

        # Save chat history
        chatbot.save_chat_history()

        # Return response
        return jsonify({
            'message': response['answer'],
            # 'sources': [
            #     {
            #         'file': doc.metadata.get('source_file', 'Unknown'),
            #         'page': doc.metadata.get('page', 'Unknown')
            #     }
            #     for doc in response.get('source_documents', [])
            # ]
        })

    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear chat history"""
    try:
        chatbot.clear_history()
        # Start a new session
        new_session_id = chatbot.start_session()
        return jsonify({
            'status': 'success',
            'session_id': new_session_id
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    # In your main script or during initialization
    chatbot = PDFChatbot(storage_directory="./storage_directory")
    # chatbot.check_vectorstore_health()
    app.run(debug=True)