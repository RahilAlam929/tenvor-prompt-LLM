import { NextResponse } from "next/server";

type Context = { params: Promise<{ id: string }> };

const base = () =>
  (process.env.TENVOR_API_URL || "http://127.0.0.1:8002").replace(/\/$/, "");

export async function GET(_request: Request, context: Context) {
  try {
    const { id } = await context.params;
    const response = await fetch(
      `${base()}/v1/conversations/${encodeURIComponent(id)}`,
      { cache: "no-store" },
    );
    return new NextResponse(await response.text(), {
      status: response.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return NextResponse.json({ error: "Cannot connect to Tenvor API." }, { status: 502 });
  }
}

export async function PATCH(request: Request, context: Context) {
  try {
    const { id } = await context.params;
    const response = await fetch(
      `${base()}/v1/conversations/${encodeURIComponent(id)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(await request.json()),
        cache: "no-store",
      },
    );
    return new NextResponse(await response.text(), {
      status: response.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return NextResponse.json({ error: "Cannot connect to Tenvor API." }, { status: 502 });
  }
}

export async function DELETE(_request: Request, context: Context) {
  try {
    const { id } = await context.params;
    const response = await fetch(
      `${base()}/v1/conversations/${encodeURIComponent(id)}`,
      { method: "DELETE", cache: "no-store" },
    );
    return new NextResponse(await response.text(), {
      status: response.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return NextResponse.json({ error: "Cannot connect to Tenvor API." }, { status: 502 });
  }
}
