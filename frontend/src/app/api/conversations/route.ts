import { NextResponse } from "next/server";

const base = () =>
  (process.env.TENVOR_API_URL || "http://127.0.0.1:8002").replace(/\/$/, "");

export async function GET() {
  try {
    const response = await fetch(`${base()}/v1/conversations`, {
      cache: "no-store",
    });
    return new NextResponse(await response.text(), {
      status: response.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return NextResponse.json(
      { error: "Cannot connect to Tenvor API on port 8002." },
      { status: 502 },
    );
  }
}

export async function POST(request: Request) {
  try {
    const response = await fetch(`${base()}/v1/conversations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(await request.json()),
      cache: "no-store",
    });
    return new NextResponse(await response.text(), {
      status: response.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return NextResponse.json(
      { error: "Cannot connect to Tenvor API on port 8002." },
      { status: 502 },
    );
  }
}
