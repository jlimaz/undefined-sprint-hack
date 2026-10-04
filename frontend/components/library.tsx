"use client";

import { cn } from "@/lib/utils";
import {
  FileJsonIcon,
  FileSpreadsheetIcon,
  FileTextIcon,
  SearchIcon,
  XIcon,
} from "lucide-react";
import { useMemo, useRef, useState, type DragEvent } from "react";

type LibraryFile = {
  id: string;
  name: string;
  size: number;
  type: string;
  addedAt: number;
  file: File;
};

type Filter = "all" | "pdf" | "json";
type Kind = "pdf" | "json" | "docs" | "data" | "other";

const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "pdf", label: "PDF" },
  { id: "json", label: "JSON" },
];

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatAdded(timestamp: number) {
  const date = new Date(timestamp);
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startOfDate = new Date(
    date.getFullYear(),
    date.getMonth(),
    date.getDate(),
  );
  const days = Math.round(
    (startOfToday.getTime() - startOfDate.getTime()) / 86_400_000,
  );
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function kindOf(file: Pick<LibraryFile, "name" | "type">): Kind {
  const name = file.name.toLowerCase();
  const type = file.type.toLowerCase();
  if (type === "application/pdf" || name.endsWith(".pdf")) return "pdf";
  if (type.includes("json") || name.endsWith(".json")) return "json";
  if (
    /\.(csv|tsv|xls|xlsx|xml)$/.test(name) ||
    type.includes("csv") ||
    type.includes("spreadsheet") ||
    type.includes("excel")
  ) {
    return "data";
  }
  if (
    /\.(doc|docx|txt|md|rtf|pages)$/.test(name) ||
    type.includes("word") ||
    type.startsWith("text/")
  ) {
    return "docs";
  }
  return "other";
}

function FileMark({ kind }: { kind: Kind }) {
  const tone = {
    pdf: "bg-red-500 text-white",
    json: "bg-amber-500 text-white",
    docs: "bg-blue-500 text-white",
    data: "bg-emerald-500 text-white",
    other: "bg-muted text-muted-foreground",
  }[kind];
  const Icon =
    kind === "json"
      ? FileJsonIcon
      : kind === "data"
        ? FileSpreadsheetIcon
        : FileTextIcon;
  return (
    <span
      className={cn(
        "flex size-7 shrink-0 items-center justify-center rounded-md",
        tone,
      )}
    >
      <Icon className="size-3.5" />
    </span>
  );
}

export function Library() {
  const inputRef = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);
  const [files, setFiles] = useState<LibraryFile[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [dragOver, setDragOver] = useState(false);

  const addFiles = (list: FileList | File[]) => {
    const next = Array.from(list).map((file) => ({
      id: crypto.randomUUID(),
      name: file.name,
      size: file.size,
      type: file.type,
      addedAt: Date.now(),
      file,
    }));
    if (next.length === 0) return;
    setFiles((current) => [...next, ...current]);
  };

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return files.filter((file) => {
      if (filter !== "all" && kindOf(file) !== filter) return false;
      if (needle && !file.name.toLowerCase().includes(needle)) return false;
      return true;
    });
  }, [files, filter, query]);

  const toggle = (id: string) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const remove = (id: string) => {
    setFiles((current) => current.filter((file) => file.id !== id));
    setSelected((current) => {
      if (!current.has(id)) return current;
      const next = new Set(current);
      next.delete(id);
      return next;
    });
  };

  const onDragEnter = (event: DragEvent<HTMLElement>) => {
    if (!event.dataTransfer.types.includes("Files")) return;
    event.preventDefault();
    dragDepth.current += 1;
    setDragOver(true);
  };

  const onDragOver = (event: DragEvent<HTMLElement>) => {
    if (!event.dataTransfer.types.includes("Files")) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  };

  const onDragLeave = (event: DragEvent<HTMLElement>) => {
    if (!event.dataTransfer.types.includes("Files")) return;
    event.preventDefault();
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDragOver(false);
  };

  const onDrop = (event: DragEvent<HTMLElement>) => {
    if (!event.dataTransfer.types.includes("Files")) return;
    event.preventDefault();
    dragDepth.current = 0;
    setDragOver(false);
    addFiles(event.dataTransfer.files);
  };

  return (
    <aside
      className="bg-sidebar flex h-[22rem] shrink-0 flex-col gap-3 border-b px-3 py-4 md:h-full md:w-[21.6rem] md:border-r md:border-b-0 md:px-3.5"
      onDragEnter={onDragEnter}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <h2 className="shrink-0 px-1 text-base font-semibold tracking-tight">
        Library
      </h2>

      <label className="relative block shrink-0">
        <SearchIcon className="text-muted-foreground pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2" />
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search files"
          aria-label="Search files"
          className="placeholder:text-muted-foreground focus-visible:ring-ring/50 h-10 w-full rounded-lg border bg-transparent pr-3 pl-8 text-sm outline-none transition-colors focus-visible:ring-1"
        />
      </label>

      <div role="tablist" aria-label="File types" className="flex shrink-0 gap-1.5">
        {FILTERS.map((item) => {
          const active = filter === item.id;
          return (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => setFilter(item.id)}
              className={cn(
                "h-8 flex-1 rounded-lg text-sm transition-colors",
                active
                  ? "bg-secondary text-foreground"
                  : "bg-muted/60 text-muted-foreground hover:text-foreground",
              )}
            >
              {item.label}
            </button>
          );
        })}
      </div>

      <div
        className={cn(
          "flex min-h-0 flex-1 flex-col gap-3",
          files.length === 0 && "group/empty",
        )}
      >
      <ul
        role="listbox"
        aria-multiselectable="true"
        aria-label="Library files"
        className="flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto px-0.5 py-0.5"
      >
        {visible.length === 0 ? (
          <li className="flex flex-1 items-center justify-center px-2">
            <span
              className={cn(
                "text-muted-foreground text-sm transition-all duration-300",
                files.length === 0 &&
                  "group-hover/empty:text-foreground/80 group-hover/empty:-translate-y-1",
              )}
            >
              {files.length === 0 ? "No files yet" : "No matching files"}
            </span>
          </li>
        ) : (
          visible.map((item) => {
            const isSelected = selected.has(item.id);
            return (
              <li key={item.id} className="group relative">
                <button
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  title={item.name}
                  onClick={() => toggle(item.id)}
                  className={cn(
                    "flex w-full items-center gap-2.5 rounded-lg border px-2 py-2 text-left",
                    isSelected
                      ? "border-blue-500/70 bg-blue-500/15"
                      : "border-transparent hover:bg-muted/50",
                  )}
                >
                  <FileMark kind={kindOf(item)} />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm">{item.name}</span>
                    <span className="text-muted-foreground block truncate text-xs">
                      {formatSize(item.size)} · {formatAdded(item.addedAt)}
                    </span>
                  </span>
                  <span
                    aria-hidden
                    className={cn(
                      "size-4 shrink-0 rounded-full border",
                      isSelected
                        ? "border-blue-500 bg-blue-500 shadow-[inset_0_0_0_3px_var(--color-sidebar)]"
                        : "border-muted-foreground/40",
                    )}
                  />
                </button>
                <button
                  type="button"
                  aria-label={`Remove ${item.name}`}
                  onClick={() => remove(item.id)}
                  className="text-muted-foreground hover:text-foreground absolute top-1/2 right-8 flex size-5 -translate-y-1/2 items-center justify-center rounded opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
                >
                  <XIcon className="size-3.5" />
                </button>
              </li>
            );
          })
        )}
      </ul>

      <div
        className={cn(
          "text-muted-foreground flex shrink-0 items-center gap-2 rounded-lg border border-dashed px-3 py-2.5 text-sm transition-colors duration-300",
          dragOver
            ? "border-foreground/50 bg-muted/30"
            : "border-foreground/25",
          files.length === 0 &&
            "group-hover/empty:border-foreground/45 group-hover/empty:bg-muted/20",
        )}
      >
        <FileTextIcon
          className={cn(
            "size-3.5 shrink-0 transition-transform duration-300",
            files.length === 0 && "group-hover/empty:-translate-y-0.5",
          )}
        />
        <p>
          Drop files anywhere, or{" "}
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="text-blue-400 underline-offset-2 hover:underline"
          >
            browse
          </button>
        </p>
      </div>
      </div>
      <input
        ref={inputRef}
        type="file"
        multiple
        className="sr-only"
        onChange={(event) => {
          if (event.target.files) addFiles(event.target.files);
          event.target.value = "";
        }}
      />
    </aside>
  );
}
