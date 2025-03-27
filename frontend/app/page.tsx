"use client";

import type React from "react";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Upload, ImageIcon, Send, Loader2 } from "lucide-react";
import Image from "next/image";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// Simulated chat messages
const SIMULATED_RESPONSES = [
  "This appears to be a beautiful landscape photo. The colors are vibrant and the composition is excellent.",
  "I notice there are several interesting elements in this image. The lighting creates a nice atmosphere.",
  "Based on what I can see, this image has great detail. The subject is well-focused and stands out clearly.",
  "The perspective in this image is quite unique. It offers an interesting viewpoint that draws the viewer in.",
  "I can see why you'd be interested in this image. It has a compelling visual story that captures attention.",
];

type Message = {
  id: string;
  content: string;
  role: "user" | "assistant";
};

export default function ImageChatPage() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [imageDescription, setImageDescription] = useState<string | null>(null);
  const [showChat, setShowChat] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const objectUrl = URL.createObjectURL(file);
    setImageUrl(objectUrl);
    setImageDescription(null);
    setShowChat(false);
    setMessages([]);
  };

  const simulateImageAnalysis = () => {
    setIsAnalyzing(true);

    // Simulate API delay
    setTimeout(() => {
      const simulatedDescription =
        "This image shows a stunning landscape with mountains in the background and a lake in the foreground. The sky has beautiful cloud formations, and the colors are vibrant with blues, greens, and hints of orange from the setting sun. The composition creates a sense of depth and tranquility, making it an excellent example of nature photography.";

      setImageDescription(simulatedDescription);
      setShowChat(true);
      setIsAnalyzing(false);
    }, 2000);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!input.trim()) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      role: "user",
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Simulate AI response delay
    setTimeout(() => {
      const randomIndex = Math.floor(
        Math.random() * SIMULATED_RESPONSES.length
      );
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: SIMULATED_RESPONSES[randomIndex],
        role: "assistant",
      };

      setMessages((prev) => [...prev, aiResponse]);
      setIsLoading(false);
    }, 1500);
  };

  return (
    <div className="container mx-auto py-8 px-4">
      <Card className="w-full max-w-full mx-auto">
        <CardHeader className="flex justify-between items-center">
          <CardTitle className="text-4xl text-center text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-blue-500 to-purple-500 font-bold">
            AI Image Analysis & Chat
          </CardTitle>
          <Select>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Selecione o modelo" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectLabel>Modelo</SelectLabel>
                <SelectItem value="apple">Modelo 1</SelectItem>
                <SelectItem value="banana">Modelo 2</SelectItem>
                <SelectItem value="banana">Modelo 1 + Modelo 2</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Image Upload Section */}
          <div className="space-y-4">
            <div className="flex items-center justify-center">
              <label
                htmlFor="image-upload"
                className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 dark:hover:bg-gray-800 dark:bg-gray-900 border-gray-300 dark:border-gray-700 transition-all"
              >
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <Upload className="w-10 h-10 mb-3 text-gray-400" />
                  <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
                    <span className="font-semibold">Click to upload</span> or
                    drag and drop
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    PNG or JPG (MAX. 10MB)
                  </p>
                </div>
                <input
                  id="image-upload"
                  type="file"
                  className="hidden"
                  accept="image/*"
                  onChange={handleImageUpload}
                />
              </label>
            </div>

            {imageUrl && (
              <div className="flex flex-col items-center space-y-4">
                <div className="flex justify-evenly w-full h-auto gap-4">
                  <Image
                    src={imageUrl || "/placeholder.svg"}
                    alt="Uploaded image"
                    width={350}
                    height={350}
                    className="object-contain rounded-lg shadow-md"
                  />
                  <Image
                    src={imageUrl || "/placeholder.svg"}
                    alt="Uploaded image"
                    width={350}
                    height={350}
                    className="object-contain rounded-lg shadow-md"
                  />
                </div>
                <div className="flex gap-2.5">
                  <Select>
                    <SelectTrigger className="w-[180px]">
                      <SelectValue placeholder="Selecione o modelo" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectLabel>Modelo</SelectLabel>
                        <SelectItem value="apple">Nosso</SelectItem>
                        <SelectItem value="banana">Pre-feito</SelectItem>
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                  <Button
                    onClick={simulateImageAnalysis}
                    disabled={isAnalyzing}
                    className="flex items-center space-x-2 bg-gradient-to-r from-blue-400 via-blue-500 to-purple-500 text-white"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Analyzing...</span>
                      </>
                    ) : (
                      <>
                        <ImageIcon className="w-4 h-4" />
                        <span>Analyze Image</span>
                      </>
                    )}
                  </Button>
                </div>
              </div>
            )}
          </div>

          {/* Image Description Section */}
          {imageDescription && (
            <div className="space-y-2">
              <h3 className="text-2xl font-semibold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">
                Image Description:
              </h3>
              <div className="p-4 rounded-lg bg-gray-100 dark:bg-gray-800 shadow-md">
                <p>{imageDescription}</p>
              </div>
            </div>
          )}

          {/* Chat Section */}
          {showChat && (
            <div className="space-y-4">
              <div className="flex justify-between">
                <h3 className="text-2xl font-semibold text-transparent text-gradient bg-clip-text bg-gradient-to-r from-blue-400 via-blue-500 to-purple-500">
                  Chat with AI about this image:
                </h3>
                <Select>
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Selecione o modelo" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      <SelectLabel>Modelo</SelectLabel>
                      <SelectItem value="apple">Nosso</SelectItem>
                      <SelectItem value="banana">Pre-feito</SelectItem>
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </div>
              <div className="h-80 overflow-y-auto p-4 rounded-lg space-y-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${
                      message.role === "user" ? "justify-end" : "justify-start"
                    }`}
                  >
                    <div
                      className={`max-w-[80%] p-3 rounded-lg ${
                        message.role === "user"
                          ? "bg-primary text-primary-foreground"
                          : "bg-secondary text-secondary-foreground"
                      }`}
                    >
                      {message.content}
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="max-w-[80%] p-3 rounded-lg bg-secondary text-secondary-foreground">
                      <Loader2 className="w-4 h-4 animate-spin" />
                    </div>
                  </div>
                )}
              </div>
              <form onSubmit={handleSubmit} className="flex space-x-2">
                <Textarea
                  value={input}
                  onChange={handleInputChange}
                  placeholder="Ask about the image..."
                  className="flex-1 resize-none"
                  rows={2}
                />
                <Button
                  type="submit"
                  disabled={isLoading || !input.trim()}
                  className="self-end bg-gradient-to-r from-blue-400 via-blue-500 to-purple-500 text-white"
                >
                  <Send className="w-4 h-4" />
                </Button>
              </form>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
