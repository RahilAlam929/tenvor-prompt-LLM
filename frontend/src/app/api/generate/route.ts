import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();

    if (
      typeof body.conversationId !== "string" ||
      typeof body.message !== "string" ||
      !body.message.trim()
    ) {
      return NextResponse.json(
        { error: "A conversation ID and message are required." },
        { status: 400 },
      );
    }

    const base = (
      process.env.TENVOR_API_URL || "http://127.0.0.1:8002"
    ).replace(/\/$/, "");

    const response = await fetch(
      `${base}/v1/conversations/${encodeURIComponent(body.conversationId)}/chat`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: body.message,
          provider: body.provider || "ollama",
          model: body.model || "llama3.2:3b",
        }),
        cache: "no-store",
      },
    );

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { error: data.detail || data.error || "AI response generate nahi hua." },
        { status: response.status },
      );
    }

    return NextResponse.json({
      reply: data.assistant_message.content,
      provider: data.provider,
      model: data.model,
      message: data.assistant_message,
    });
  } catch {
    return NextResponse.json(
      { error: "Backend connect nahi ho raha. Port 8002 aur selected provider check karo." },
      { status: 502 },
    );
  }
}
