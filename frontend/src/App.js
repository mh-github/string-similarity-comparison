import React, { useState } from 'react';
import './App.css';

function App() {
  const [inputText1, setInputText1] = useState('');
  const [inputText2, setInputText2] = useState('');
  const [response, setResponse] = useState(null); // Changed to null for initial state to handle object response

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch('/api/submit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ inputText1, inputText2 }),
      });

      const data = await response.json();
      setResponse(data); // Expecting data to be an object with the required fields
    } catch (error) {
      console.error('Error submitting data:', error);
    }
  };

  return (
    <div>
      <h1>String Similarity Comparison</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label>
            Enter first string:
            <input
              type="text"
              value={inputText1}
              onChange={(e) => setInputText1(e.target.value)}
            />
          </label>
        </div>
        <div> {/* New div wrapping the second label and input field */}
          <label>
            Enter second string:
            <input
              type="text"
              value={inputText2}
              onChange={(e) => setInputText2(e.target.value)}
            />
          </label>
        </div>
        <button type="submit">Submit</button>
      </form>
      {response && (
        <div>
          <h2>Response from Backend:</h2>
          <table>
            <thead>
              <tr>
                <th>String 1</th>
                <th>String 2</th>
                <th>Cosine Similarity</th>
                <th>Gensim Similarity</th>
                <th>spaCy Similarity</th>
                <th>Google Similarity</th>
                <th>Hugging Face Similarity</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{response.string1}</td>
                <td>{response.string2}</td>
                <td>{response.cosine_similarity}</td>
                <td>{response.gensim_similarity}</td>
                <td>{response.spacy_similarity}</td>
                <td>{response.google_similarity}</td>
                <td>{response.hugging_face_similarity}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default App;
