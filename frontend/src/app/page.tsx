"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { signIn, signOut, useSession } from "next-auth/react";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  provider?: string | null;
  model?: string | null;
};

type Conversation = {
  id: string;
  title: string;
  updated_at: string;
  message_count?: number;
};

const MODELS: Record<string, string[]> = {
  ollama: ["llama3.2:3b"],
  tenvor: ["tenvor-gpt"],
  openai: ["gpt-4o-mini"],
};

export default function Home() {
  const { data: session, status: authStatus } = useSession();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [provider, setProvider] = useState("ollama");
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    const saved = localStorage.getItem("tenvor-theme");
    if (saved === "light" || saved === "dark") {
      const timer = window.setTimeout(() => setTheme(saved), 0);
      return () => window.clearTimeout(timer);
    }
  }, []);

  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("tenvor-theme", next);
  }
  const [model, setModel] = useState("llama3.2:3b");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const bottom = useRef<HTMLDivElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);

  const refreshConversations = useCallback(async () => {
    try {
      const response = await fetch("/api/conversations", { cache: "no-store" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "History load failed");
      setConversations(data.conversations || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "History load failed");
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refreshConversations();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [refreshConversations]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  async function openConversation(id: string) {
    if (loading) return;
    setError("");

    try {
      const response = await fetch(`/api/conversations/${encodeURIComponent(id)}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || "Could not open chat");

      setActiveId(id);
      setMessages(data.messages || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not open chat");
    }
  }

  function newChat() {
    if (loading) return;
    setActiveId(null);
    setMessages([]);
    setInput("");
    setError("");
    textarea.current?.focus();
  }

  async function renameConversation(conversation: Conversation) {
    const title = window.prompt("New conversation name:", conversation.title);
    if (title === null || !title.trim()) return;

    try {
      const response = await fetch(
        `/api/conversations/${encodeURIComponent(conversation.id)}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: title.trim() }),
        },
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || "Rename failed");
      await refreshConversations();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Rename failed");
    }
  }

  async function deleteConversation(conversation: Conversation) {
    if (!window.confirm(`Delete "${conversation.title}" and its messages?`)) return;

    try {
      const response = await fetch(
        `/api/conversations/${encodeURIComponent(conversation.id)}`,
        { method: "DELETE" },
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || "Delete failed");

      if (activeId === conversation.id) newChat();
      await refreshConversations();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
    }
  }

  async function send(text = input) {
    const message = text.trim();
    if (!message || loading) return;

    setInput("");
    setError("");
    setLoading(true);

    const optimisticUser: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: message,
    };
    setMessages((old) => [...old, optimisticUser]);

    let conversationId = activeId;

    try {
      if (!conversationId) {
        const createResponse = await fetch("/api/conversations", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: "New chat" }),
        });
        const created = await createResponse.json();
        if (!createResponse.ok) {
          throw new Error(created.detail || created.error || "Could not create chat");
        }

        conversationId = created.id;
        setActiveId(conversationId);
      }

      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversationId,
          message,
          provider,
          model,
        }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || data.detail || "AI request failed");

      setMessages((old) => [
        ...old,
        {
          ...data.message,
          id: data.message?.id || crypto.randomUUID(),
          role: "assistant",
          content: data.reply || "No response received.",
        },
      ]);

      await refreshConversations();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
      textarea.current?.focus();
    }
  }

  function changeProvider(nextProvider: string) {
    setProvider(nextProvider);
    setModel(MODELS[nextProvider]?.[0] || "");
  }

  return (
    <main data-theme={theme} className="flex h-dvh overflow-hidden bg-[#090b10] text-[#eef0f6]">
      <aside className="hidden w-[280px] shrink-0 flex-col border-r border-white/[0.07] bg-[#0d1017] md:flex">
        <div className="flex items-center gap-3 px-5 py-6">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-400 to-indigo-600 text-lg font-bold shadow-lg shadow-violet-950/40">
            T
          </div>
          <div>
            <div className="text-base font-semibold tracking-tight">Tenvor</div>
            <div className="text-[11px] text-slate-500">PROMPT WORKSPACE</div>
          </div>
        </div>

        <div className="px-3">
          <button
            onClick={newChat}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-3 text-sm font-medium transition hover:bg-white/[0.08]"
          >
            <span className="text-lg">＋</span> New conversation
          </button>
        </div>

        <div className="mt-7 flex items-center justify-between px-5">
          <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
            Recent chats
          </span>
          <button
            onClick={() => void refreshConversations()}
            title="Refresh history"
            className="rounded-lg px-2 py-1 text-slate-500 hover:bg-white/5 hover:text-white"
          >
            ↻
          </button>
        </div>

        <nav className="mt-3 flex-1 space-y-1 overflow-y-auto px-2 pb-4">
          {conversations.map((conversation) => (
            <div
              key={conversation.id}
              className={`group flex items-center gap-1 rounded-xl ${
                activeId === conversation.id
                  ? "bg-violet-400/10 text-white"
                  : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-200"
              }`}
            >
              <button
                onClick={() => void openConversation(conversation.id)}
                className="min-w-0 flex-1 px-3 py-3 text-left text-sm"
                title={conversation.title}
              >
                <span className="block truncate">{conversation.title}</span>
                <span className="mt-1 block text-[10px] text-slate-600">
                  {conversation.message_count ?? 0} messages
                </span>
              </button>
              <div className="flex pr-1 opacity-0 transition group-hover:opacity-100">
                <button
                  onClick={() => void renameConversation(conversation)}
                  title="Rename"
                  className="rounded-md p-1.5 text-xs hover:bg-white/10"
                >
                  ✎
                </button>
                <button
                  onClick={() => void deleteConversation(conversation)}
                  title="Delete"
                  className="rounded-md p-1.5 text-xs text-rose-300 hover:bg-rose-400/10"
                >
                  ×
                </button>
              </div>
            </div>
          ))}

          {conversations.length === 0 && (
            <p className="px-3 py-5 text-xs leading-5 text-slate-600">
              Your saved conversations will appear here.
            </p>
          )}
        </nav>

        <div className="mt-auto border-t border-white/[0.07] p-4">
          {authStatus === "loading" ? (
            <div className="rounded-xl border border-white/[0.07] p-3 text-xs text-slate-500">
              Checking account...
            </div>
          ) : session?.user ? (
            <div className="rounded-xl border border-white/[0.07] bg-white/[0.035] p-3">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-violet-400 to-indigo-600 text-sm font-semibold text-white">
                  {(session.user.name || session.user.email || "U").charAt(0).toUpperCase()}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium text-slate-200">
                    {session.user.name || "Signed in"}
                  </div>
                  <div className="truncate text-[11px] text-slate-500">
                    {session.user.email}
                  </div>
                </div>
              </div>
              <button
                onClick={() => void signOut({ callbackUrl: "/" })}
                className="mt-3 w-full rounded-lg border border-white/10 px-3 py-2 text-xs text-slate-300 transition hover:bg-white/[0.06]"
              >
                Sign out
              </button>
            </div>
          ) : (
            <button
              onClick={() => void signIn("google")}
              className="flex w-full items-center justify-center gap-3 rounded-xl border border-white/10 bg-white/[0.05] px-3 py-3 text-sm font-medium text-slate-100 transition hover:bg-white/[0.1]"
            >
              <svg viewBox="0 0 48 48" aria-hidden="true" className="h-5 w-5">
                <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5Z" transform="translate(0 4) scale(1 .83)"/>
                <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.74 7.18l7.68 5.96c4.48-4.14 7.1-10.24 7.1-17.61Z"/>
                <path fill="#FBBC05" d="M10.53 28.59A14.4 14.4 0 0 1 9.75 24c0-1.59.27-3.13.76-4.59l-7.98-6.19A23.9 23.9 0 0 0 0 24c0 3.87.93 7.53 2.56 10.78l7.97-6.19Z" transform="translate(0 4) scale(1 .83)"/>
                <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.9-5.84l-7.68-5.96c-2.13 1.43-4.85 2.28-8.22 2.28-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 43.62 14.62 48 24 48Z" transform="translate(0 4) scale(1 .83)"/>
              </svg>
              Continue with Google
            </button>
          )}
          <p className="mt-3 text-center text-[10px] text-slate-600">
            Your AI workspace
          </p>
        </div>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-[70px] shrink-0 items-center justify-between border-b border-white/[0.07] px-4 sm:px-7">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-violet-400 to-indigo-600 font-bold md:hidden">
              T
            </div>
            <div>
              <div className="text-sm font-semibold">AI Workspace</div>
              <div className="mt-0.5 text-[11px] text-slate-500">
                Think, build, and iterate.
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={toggleTheme}
              aria-label="Toggle color theme"
              title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
              className="rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 text-xs transition hover:bg-white/10"
            >
              {theme === "dark" ? "☀ Light" : "☾ Dark"}
            </button>
            <span className="hidden text-[11px] text-slate-500 sm:inline">Provider</span>
            <select
              value={provider}
              onChange={(event) => changeProvider(event.target.value)}
              disabled={loading}
              aria-label="AI provider"
              className="max-w-[125px] rounded-xl border border-white/10 bg-[#141823] px-3 py-2 text-xs outline-none focus:border-violet-400/60"
            >
              <option value="ollama">Ollama</option>
              <option value="tenvor">TenvorGPT</option>
              <option value="openai">OpenAI</option>
            </select>
            <select
              value={model}
              onChange={(event) => setModel(event.target.value)}
              disabled={loading}
              aria-label="AI model"
              className="max-w-[140px] rounded-xl border border-white/10 bg-[#141823] px-3 py-2 text-xs outline-none focus:border-violet-400/60"
            >
              {(MODELS[provider] || [model]).map((item) => (
                <option key={item} value={item}>{item}</option>
              ))}
            </select>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="flex min-h-full flex-col items-center justify-center px-5 pb-12 pt-10">
              <div className="mb-6 flex h-[68px] w-[68px] items-center justify-center rounded-[24px] border border-violet-300/20 bg-gradient-to-br from-violet-400/15 to-indigo-500/10 text-3xl shadow-2xl shadow-violet-950/20">
                ✳
              </div>
              <p className="mb-3 text-[11px] font-semibold uppercase tracking-[0.24em] text-violet-300">
                Your AI workspace
              </p>
              <h1 className="max-w-xl text-center text-3xl font-semibold tracking-tight sm:text-5xl">
                What are we building <span className="text-violet-300">today?</span>
              </h1>
              <p className="mt-4 max-w-md text-center text-sm leading-6 text-slate-500">
                Ask a question, explore an idea, or work through your next coding challenge.
              </p>

              <div className="mt-9 grid w-full max-w-2xl gap-3 sm:grid-cols-3">
                {[
                  ["Build an AI project", "Plan architecture and implementation"],
                  ["Debug my code", "Find bugs and explain the fix"],
                  ["Improve my prompt", "Make instructions clearer and stronger"],
                ].map(([title, description]) => (
                  <button
                    key={title}
                    onClick={() => void send(title)}
                    className="rounded-2xl border border-white/[0.09] bg-white/[0.025] p-4 text-left transition hover:-translate-y-0.5 hover:border-violet-300/30 hover:bg-violet-300/[0.05]"
                  >
                    <span className="block text-sm font-medium">{title}</span>
                    <span className="mt-2 block text-xs leading-5 text-slate-500">{description}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl px-4 py-8 sm:px-8 sm:py-10">
              {messages.map((message) => (
                <div key={message.id} className="mb-8">
                  {message.role === "user" ? (
                    <div className="ml-auto w-fit max-w-[88%] whitespace-pre-wrap rounded-2xl rounded-br-md border border-white/[0.08] bg-[#191e2a] px-4 py-3.5 text-sm leading-7 sm:px-5">
                      {message.content}
                    </div>
                  ) : (
                    <div className="flex gap-3 sm:gap-4">
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-violet-400 to-indigo-600 text-xs font-bold shadow-lg shadow-violet-950/30">
                        T
                      </div>
                      <div className="min-w-0 flex-1 pt-1">
                        <div className="mb-2 flex items-center gap-2">
                          <span className="text-xs font-semibold text-slate-200">Tenvor</span>
                          {message.model && (
                            <span className="rounded-md border border-white/[0.07] px-1.5 py-0.5 text-[10px] text-slate-500">
                              {message.model}
                            </span>
                          )}
                        </div>
                        <div className="whitespace-pre-wrap break-words text-sm leading-7 text-slate-200">
                          {message.content}
                        </div>
                        <button
                          onClick={() => void navigator.clipboard.writeText(message.content)}
                          className="mt-3 rounded-lg px-2 py-1 text-[11px] text-slate-500 transition hover:bg-white/5 hover:text-slate-200"
                        >
                          Copy response
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
              {loading && (
                <div className="flex items-center gap-3 text-sm text-slate-400">
                  <div className="h-2 w-2 animate-pulse rounded-full bg-violet-300" />
                  Tenvor is thinking…
                </div>
              )}
              <div ref={bottom} />
            </div>
          )}
        </div>

        <div className="shrink-0 px-3 pb-3 pt-2 sm:px-6 sm:pb-5">
          <div className="mx-auto max-w-3xl">
            <div className="rounded-[22px] border border-white/[0.12] bg-[#121620] p-3 shadow-2xl shadow-black/20 transition focus-within:border-violet-300/40 sm:p-4">
              <textarea
                ref={textarea}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void send();
                  }
                }}
                placeholder="Message Tenvor…"
                rows={2}
                maxLength={10000}
                className="max-h-40 min-h-12 w-full resize-y bg-transparent px-1 py-1 text-sm leading-6 outline-none placeholder:text-slate-600"
              />
              <div className="flex items-center justify-between pt-2">
                <span className="text-[10px] text-slate-600">
                  Enter to send · Shift + Enter for new line
                </span>
                <button
                  onClick={() => void send()}
                  disabled={!input.trim() || loading}
                  aria-label="Send message"
                  className="flex h-9 w-9 items-center justify-center rounded-xl bg-violet-400 text-lg font-semibold text-[#100d1b] transition hover:bg-violet-300 disabled:cursor-not-allowed disabled:bg-white/10 disabled:text-slate-600"
                >
                  ↑
                </button>
              </div>
            </div>
            {error && (
              <div className="mt-2 rounded-xl border border-rose-400/20 bg-rose-400/[0.06] px-3 py-2 text-xs text-rose-300">
                {error}
              </div>
            )}
            <p className="mt-3 text-center text-[10px] text-slate-600">
              AI responses can be inaccurate. Verify important information.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}
