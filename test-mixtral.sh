#!/bin/bash

generate_text() {
    local question="$1"
    local response
    response=$(curl -s http://localhost:11434/api/chat -d '{
        "model": "mixtral",
        "messages": [
            {
                "role": "user",
                "content": "'"${question}"'"
            }
        ],
	"stream": false
    }' | jq -r '.message.content')

    # Change color to green
    echo -e "\033[0;32m$response\033[0m"
}

while true; do
    read -p "Enter your question: " question
    generate_text "$question"
done
