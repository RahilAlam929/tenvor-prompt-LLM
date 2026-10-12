import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const base =
      process.env.TENVOR_API_URL || "http://127.0.0.1:8002";

    const response = await fetch(`${base}/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: body.message,
        model: "llama3.2:3b",
      }),
      cache: "no-store",
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { error: data.detail || "AI response generate nahi hua." },
        { status: response.status }
      );
    }

    return NextResponse.json({
      reply: data.result,
      provider: data.provider,
      model: data.model,
    });
  } catch {
    return NextResponse.json(
      { error: "Backend connect nahi ho raha. Port 8002 aur Ollama check karo." },
      { status: 502 }
    );
  }
}
