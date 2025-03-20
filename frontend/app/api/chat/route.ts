import { openai } from "@ai-sdk/openai"
import { streamText } from "ai"

export const maxDuration = 30 // Allow streaming responses up to 30 seconds

export async function POST(request: Request) {
  const { messages } = await request.json()

  const result = streamText({
    model: openai("gpt-4o"),
    messages,
  })

  return result.toDataStreamResponse()
}

