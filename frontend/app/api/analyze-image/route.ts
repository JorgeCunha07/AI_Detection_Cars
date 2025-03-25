import { type NextRequest, NextResponse } from "next/server"
import { openai } from "@ai-sdk/openai"
import { generateText } from "ai"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const imageFile = formData.get("image") as File

    if (!imageFile) {
      return NextResponse.json({ error: "No image file provided" }, { status: 400 })
    }

    // Convert the file to a buffer
    const bytes = await imageFile.arrayBuffer()
    const buffer = Buffer.from(bytes)

    // Convert to base64
    // const base64Image = buffer.toString("base64")

    // Generate description using OpenAI's vision capabilities
    const { text } = await generateText({
      model: openai("gpt-4o"),
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Describe this image in detail." },
            {
              type: "image",
              image: buffer,
            },
          ],
        },
      ],
    })

    return NextResponse.json({ description: text })
  } catch (error) {
    console.error("Error analyzing image:", error)
    return NextResponse.json({ error: "Failed to analyze image" }, { status: 500 })
  }
}

export const config = {
  api: {
    bodyParser: false,
  },
}

