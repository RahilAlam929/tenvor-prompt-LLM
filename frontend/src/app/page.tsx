"use client";

import { useRef, useState, useEffect } from "react";

type Message = { role: "user" | "assistant"; content: string };

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(text = input) {
    const message = text.trim();
    if (!message || loading) return;

    setInput("");
    setError("");
    setMessages((old) => [...old, { role: "user", content: message }]);
    setLoading(true);

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || data.detail || "Request failed");

      setMessages((old) => [
        ...old,
        { role: "assistant", content: data.reply || "No response received." },
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function newChat() {
    setMessages([]);
    setInput("");
    setError("");
  }

  return (
    <main className="flex h-dvh bg-[#212121] text-[#ececec]">
      <aside className="hidden w-64 shrink-0 flex-col bg-[#171717] p-3 sm:flex">
        <div className="flex items-center justify-between px-2 py-3">
          <span className="text-xl font-semibold">Tenvor</span>
          <button onClick={newChat} title="New chat" className="rounded-lg p-2 hover:bg-[#303030]">
            ✎
          </button>
        </div>
        <button onClick={newChat} className="rounded-lg px-3 py-3 text-left text-sm hover:bg-[#303030]">
          ＋ New chat
        </button>
        <p className="mt-auto px-3 py-3 text-xs text-gray-500">Tenvor Prompt</p>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 items-center justify-between px-5">
          <span className="font-medium sm:hidden">Tenvor</span>
          <span className="hidden text-sm sm:block">Tenvor <span className="text-gray-500">⌄</span></span>
          <button onClick={newChat} className="rounded-lg px-3 py-2 text-sm hover:bg-[#303030]">
            New chat
          </button>
        </header>

        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center px-5 pb-12">
              <h1 className="mb-8 text-center text-2xl font-semibold sm:text-3xl">
                What can I help with?
              </h1>
              <div className="flex flex-wrap justify-center gap-2">
                {[
                  "Build an AI project",
                  "Improve my prompt",
                  "Explain a coding problem",
                ].map((s) => (
                  <button key={s} onClick={() => void send(s)} className="rounded-full border border-[#424242] px-4 py-2 text-sm text-gray-300 hover:bg-[#303030]">
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl px-4 py-6 sm:px-6">
              {messages.map((m, i) => (
                <div key={i} className="mb-7">
                  {m.role === "user" ? (
                    <div className="ml-auto w-fit max-w-[85%] whitespace-pre-wrap rounded-3xl bg-[#303030] px-5 py-3 leading-7">
                      {m.content}
                    </div>
                  ) : (
                    <div className="flex gap-3 leading-7">
                      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-white text-xs font-bold text-black">T</div>
                      <div className="min-w-0 flex-1 whitespace-pre-wrap break-words">{m.content}
                        <button onClick={() => void navigator.clipboard.writeText(m.content)} className="mt-2 block rounded-lg px-2 py-1 text-xs text-gray-400 hover:bg-[#303030]">Copy</button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
              {loading && <p className="text-sm text-gray-400">Thinking...</p>}
              <div ref={bottom} />
            </div>
          )}
        </div>

        <div className="px-3 pb-3 pt-2 sm:px-6">
          <div className="mx-auto max-w-3xl rounded-[28px] border border-[#424242] bg-[#303030] p-3">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void send();
                }
              }}
              placeholder="Message Tenvor"
              rows={1}
              className="max-h-40 min-h-9 w-full resize-none bg-transparent px-2 py-1 outline-none placeholder:text-gray-400"
            />
            <div className="flex items-center justify-between px-1 pt-2">
              <span className="text-xs text-gray-500">Tenvor Prompt</span>
              <button onClick={() => void send()} disabled={!input.trim() || loading} aria-label="Send" className="flex h-9 w-9 items-center justify-center rounded-full bg-white text-xl text-black disabled:bg-[#555] disabled:text-gray-400">
                ↑
              </button>
            </div>
          </div>
          {error && <p className="mx-auto mt-2 max-w-3xl text-center text-sm text-red-400">{error}</p>}
          <p className="mt-3 text-center text-xs text-gray-500">Tenvor can make mistakes. Verify important information.</p>
        </div>
      </section>
    </main>
  );
}
