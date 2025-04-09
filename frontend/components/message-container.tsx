import Message from "@/app/types/Message";
import { ImageIcon, Loader2 } from "lucide-react";
import { useEffect, useRef } from "react";

interface MessageContainerProps {
    messages: Message[];
    isLoading: boolean;
}

export default function MessageContainer({
    messages,
    isLoading,
}: MessageContainerProps) {
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const container = containerRef.current;
        if (container) {
            container.scrollTo({
                top: container.scrollHeight,
                behavior: "smooth",
            });
        }
    }, [messages]);

    return (
        <div
            className="h-[500px] overflow-y-auto p-4 rounded-lg bg-gray-50 dark:bg-gray-900 space-y-4"
            ref={containerRef}
        >
            {messages.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                    <ImageIcon className="mx-auto h-12 w-12 mb-3 text-gray-400" />
                    <p>
                        Faça perguntas sobre a situação de trânsito mostrada na
                        imagem.
                    </p>
                    <p className="text-sm mt-2">
                        Exemplo: "O que devo fazer nesta situação?" ou "Quais
                        são os riscos aqui?"
                    </p>
                </div>
            )}

            {messages.map((message) => (
                <div
                    key={message.id}
                    className={`flex ${
                        message.role === "user"
                            ? "justify-end"
                            : "justify-start"
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
    );
}
