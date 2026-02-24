import React, { useEffect, useState, useCallback } from "react";
import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib";

interface Change {
  id: number;
  tag: "delete" | "insert" | "replace";
  old: string;
  new: string;
  context_before?: string;
  context_after?: string;
}

interface Segment {
  html: string;
  change: Change | null;
}

interface ClickableDiffProps extends ComponentProps {
  args: {
    segments: Segment[];
    height?: number;
    edited_text: string;
  };
}

// Static styles moved outside component to avoid recreation on each render
const baseButtonStyle: React.CSSProperties = {
  backgroundColor: "var(--primary-color, #ff4b4b)",
  color: "var(--background-color, #ffffff)",
  border: "1px solid var(--primary-color, #ff4b4b)",
  padding: "0.45rem 0.9rem",
  borderRadius: "0.5rem",
  cursor: "pointer",
  fontSize: "16px",
  fontWeight: "600",
  marginRight: "8px",
};

const baseUndoButtonStyle: React.CSSProperties = {
  ...baseButtonStyle,
  backgroundColor: "var(--secondary-background-color, #f0f2f6)",
  color: "var(--text-color, #262730)",
  border: "1px solid rgba(49, 51, 63, 0.2)",
};

const baseDisabledButtonStyle: React.CSSProperties = {
  ...baseUndoButtonStyle,
  cursor: "not-allowed",
  opacity: 0.5,
};

const stickyBarStyle: React.CSSProperties = {
  position: "sticky",
  top: 0,
  zIndex: 5,
  backgroundColor: "var(--background-color, #ffffff)",
  borderBottom: "1px solid rgba(49, 51, 63, 0.12)",
  padding: "8px 0",
  marginBottom: "8px",
};

const barContentStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: "8px",
  flexWrap: "wrap",
};

const toggleButtonStyle: React.CSSProperties = {
  ...baseUndoButtonStyle,
  padding: "0.35rem 0.75rem",
  fontSize: "14px",
};

const buttonGroupStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: "8px",
  flexWrap: "wrap",
};

const ClickableDiff: React.FC<ClickableDiffProps> = ({ args }) => {
  const { segments, height, edited_text } = args;
  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const [rejectedChanges, setRejectedChanges] = useState<Change[]>([]);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [editableText, setEditableText] = useState<string>(edited_text || "");
  const containerRef = React.useRef<HTMLDivElement>(null);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  // Sync editable text when new content arrives (after save/rerun)
  useEffect(() => {
    setEditableText(edited_text || "");
    setRejectedChanges([]);
    setHasUnsavedChanges(false);
  }, [edited_text]);

  useEffect(() => {
    const baseText = edited_text || "";
    const isDirty = rejectedChanges.length > 0 || editableText !== baseText;
    setHasUnsavedChanges(isDirty);
  }, [rejectedChanges, editableText, edited_text]);

  // Report height to Streamlit
  useEffect(() => {
    Streamlit.setFrameHeight();
  }, [segments, rejectedChanges, editableText]);

  const applyChangeToText = (
    text: string,
    change: Change,
    mode: "reject" | "restore",
  ): string => {
    const isReject = mode === "reject";
    const replaceOnce = (source: string, search: string, replacement: string) => {
      if (!search) {
        return source;
      }
      const idx = source.indexOf(search);
      if (idx === -1) {
        return source;
      }
      return `${source.slice(0, idx)}${replacement}${source.slice(
        idx + search.length,
      )}`;
    };

    if (change.tag === "insert") {
      if (isReject) {
        // When removing an insert, also handle adjacent spaces to avoid double spaces
        const idx = text.indexOf(change.new);
        if (idx === -1) {
          return text;
        }
        
        const before = text.slice(0, idx);
        const after = text.slice(idx + change.new.length);
        
        // If there's a space before the insert and either the insert starts a new "segment"
        // or there's a space after, remove the trailing space from before
        if (before.endsWith(" ") && (after.startsWith(" ") || after.length === 0 || change.new.startsWith("\n"))) {
          return before.slice(0, -1) + after;
        }
        
        // Otherwise just remove the inserted text
        return before + after;
      }
      const ctxBefore = change.context_before || "";
      const ctxAfter = change.context_after || "";
      if (ctxBefore && ctxAfter) {
        const anchor = `${ctxBefore} ${ctxAfter}`;
        if (text.includes(anchor)) {
          return replaceOnce(text, anchor, `${ctxBefore} ${change.new} ${ctxAfter}`);
        }
      }
      if (ctxBefore && text.includes(ctxBefore)) {
        const idx = text.indexOf(ctxBefore) + ctxBefore.length;
        return `${text.slice(0, idx)} ${change.new}${text.slice(idx)}`;
      }
      if (ctxAfter && text.includes(ctxAfter)) {
        const idx = text.indexOf(ctxAfter);
        return `${text.slice(0, idx)}${change.new} ${text.slice(idx)}`;
      }
      return `${text} ${change.new}`.trim();
    }

    if (change.tag === "replace") {
      return isReject
        ? replaceOnce(text, change.new, change.old)
        : replaceOnce(text, change.old, change.new);
    }

    if (change.tag === "delete") {
      if (isReject) {
        const ctxBefore = change.context_before || "";
        const ctxAfter = change.context_after || "";

        if (ctxBefore && ctxAfter) {
          const anchor = `${ctxBefore} ${ctxAfter}`;
          if (text.includes(anchor)) {
            // Only add spaces around change.old if it doesn't already have them
            const needsSpaceBefore = !change.old.startsWith(" ") && !change.old.startsWith("\n");
            const needsSpaceAfter = !change.old.endsWith(" ") && !change.old.endsWith("\n");
            const spaceBefore = needsSpaceBefore ? " " : "";
            const spaceAfter = needsSpaceAfter ? " " : "";
            return replaceOnce(text, anchor, `${ctxBefore}${spaceBefore}${change.old}${spaceAfter}${ctxAfter}`);
          }
        }

        if (ctxBefore && text.includes(ctxBefore)) {
          const idx = text.indexOf(ctxBefore) + ctxBefore.length;
          const needsSpace = !change.old.startsWith(" ") && !change.old.startsWith("\n");
          const space = needsSpace ? " " : "";
          return `${text.slice(0, idx)}${space}${change.old}${text.slice(idx)}`;
        }

        if (ctxAfter && text.includes(ctxAfter)) {
          const idx = text.indexOf(ctxAfter);
          const needsSpace = !change.old.endsWith(" ") && !change.old.endsWith("\n");
          const space = needsSpace ? " " : "";
          return `${text.slice(0, idx)}${change.old}${space}${text.slice(idx)}`;
        }

        // Fallback: add to end with space if needed
        const needsSpace = text.length > 0 && !text.endsWith(" ") && !change.old.startsWith(" ");
        const space = needsSpace ? " " : "";
        return `${text}${space}${change.old}`;
      }

      return replaceOnce(text, change.old, "");
    }

    return text;
  };

  // Handle click on a change span - add to rejected changes and update edit text
  const handleChangeClick = useCallback((change: Change) => {
    setRejectedChanges((prev) => {
      // Check if already rejected
      if (prev.some((c) => c.id === change.id)) {
        return prev;
      }
      return [...prev, change];
    });
    setEditableText((prev) => applyChangeToText(prev, change, "reject"));
    setHasUnsavedChanges(true);
  }, []);

  // Handle save button - send all rejected changes back to Python
  const handleSave = useCallback(() => {
    if (hasUnsavedChanges || rejectedChanges.length > 0) {
      Streamlit.setComponentValue({
        action: "save",
        rejected_changes: rejectedChanges,
        edited_text: editableText,
      });
      setRejectedChanges([]);
      setHasUnsavedChanges(false);
    }
  }, [rejectedChanges, editableText, hasUnsavedChanges]);

  // Handle undo - remove last rejected change
  const handleUndo = useCallback(() => {
    setRejectedChanges((prev) => {
      if (prev.length === 0) {
        return prev;
      }
      const last = prev[prev.length - 1];
      setEditableText((text) => applyChangeToText(text, last, "restore"));
      return prev.slice(0, -1);
    });
  }, []);

  // Build the content with clickable spans, applying rejections visually
  const renderContent = () => {
    return segments.map((segment, idx) => {
      if (segment.change === null) {
        // Context text - render as plain HTML
        return (
          <span
            key={idx}
            dangerouslySetInnerHTML={{ __html: segment.html }}
            style={{ color: "var(--text-color, inherit)" }}
          />
        );
      } else {
        // Change - make it clickable
        const change = segment.change;
        const isHovered = hoveredId === change.id;
        const isRejected = rejectedChanges.some((c) => c.id === change.id);

        // If rejected, render the reverted version
        if (isRejected) {
          return renderRejectedChange(change, idx);
        }

        return (
          <span
            key={idx}
            onClick={() => handleChangeClick(change)}
            onMouseEnter={() => setHoveredId(change.id)}
            onMouseLeave={() => setHoveredId(null)}
            style={{
              cursor: "pointer",
              position: "relative",
              display: "inline",
              outline: isHovered
                ? "2px solid var(--primary-color, #ff4b4b)"
                : "none",
              outlineOffset: "2px",
              borderRadius: "3px",
            }}
            title={getTooltip(change)}
            dangerouslySetInnerHTML={{ __html: segment.html }}
          />
        );
      }
    });
  };

  const escapeHtml = (text: string): string =>
    text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;")
      .replace(/\n/g, "<br />");

  // Render a rejected change - show what it will look like after reversion
  const renderRejectedChange = (change: Change, idx: number) => {
    let revertedHtml = "";
    const escapedOld = escapeHtml(change.old);

    switch (change.tag) {
      case "insert":
        // Insert was rejected - hide the new text but preserve spacing
        return <span key={idx}> </span>;
      case "delete":
        // Delete was rejected - show the old text without strikethrough, with spaces around it
        revertedHtml = ` ${escapedOld} `;
        break;
      case "replace":
        // Replace was rejected - show old text instead of new, with spaces around it
        revertedHtml = ` ${escapedOld} `;
        break;
    }

    return (
      <span
        key={idx}
        dangerouslySetInnerHTML={{ __html: revertedHtml }}
        style={{ color: "inherit" }}
      />
    );
  };

  const getTooltip = (change: Change): string => {
    switch (change.tag) {
      case "delete":
        return `Click to restore: "${change.old}"`;
      case "insert":
        return `Click to remove: "${change.new}"`;
      case "replace":
        return `Click to revert: "${change.new}" → "${change.old}"`;
      default:
        return "Click to reject this change";
    }
  };

  // Only height-dependent styles need to be inside the component
  const containerStyle: React.CSSProperties = {
    padding: "0.75rem 1rem",
    backgroundColor: "var(--secondary-background-color, #f0f2f6)",
    borderRadius: "8px",
    lineHeight: 1.7,
    fontFamily: "var(--font, inherit)",
    fontSize: "18px",
    color: "var(--text-color, #262730)",
    whiteSpace: "pre-wrap",
    wordWrap: "break-word",
    ...(height ? { maxHeight: `${height}px`, overflowY: "auto" as const } : {}),
  };

  const editAreaStyle: React.CSSProperties = {
    width: "calc(100% + 2rem)",
    minHeight: "320px",
    height: height ? `${height}px` : "auto",
    resize: "none",
    borderRadius: "8px",
    border: "none",
    padding: 0,
    fontFamily: "var(--font, inherit)",
    fontSize: "18px",
    lineHeight: 1.7,
    color: "var(--text-color, #262730)",
    backgroundColor: "var(--secondary-background-color, #f0f2f6)",
    boxSizing: "border-box",
    outline: "none",
  };

  return (
    <div>
      <div style={stickyBarStyle}>
        <div style={barContentStyle}>
          <div style={buttonGroupStyle}>
            <button
              onClick={() => {
                // Save scroll position before switching
                const currentScroll = isEditMode
                  ? textareaRef.current?.scrollTop || 0
                  : containerRef.current?.scrollTop || 0;
                setIsEditMode((prev) => {
                  const newMode = !prev;
                  // Restore scroll position after mode switch
                  setTimeout(() => {
                    if (newMode) {
                      if (textareaRef.current) {
                        textareaRef.current.scrollTop = currentScroll;
                      }
                    } else {
                      if (containerRef.current) {
                        containerRef.current.scrollTop = currentScroll;
                      }
                    }
                  }, 0);
                  return newMode;
                });
              }}
              style={toggleButtonStyle}
              title={isEditMode ? "Back to diff view" : "Edit text directly"}
            >
              {isEditMode ? "← Back to Diff" : "✎ Edit Text"}
            </button>
            <button
              onClick={handleSave}
              style={hasUnsavedChanges ? baseButtonStyle : baseDisabledButtonStyle}
              disabled={!hasUnsavedChanges}
              title={
                rejectedChanges.length > 0
                  ? `Apply ${rejectedChanges.length} rejection${rejectedChanges.length !== 1 ? "s" : ""}`
                  : "Save direct edits"
              }
            >
              💾 Save Changes
            </button>
            <button
              onClick={handleUndo}
              style={rejectedChanges.length > 0 ? baseUndoButtonStyle : baseDisabledButtonStyle}
              disabled={rejectedChanges.length === 0}
              title="Undo last rejection"
            >
              ↶ Undo
            </button>
          </div>
          <span
            style={{
              color: "var(--text-color, #262730)",
              opacity: 0.7,
              fontSize: "13px",
            }}
          >
            {isEditMode
              ? "Edit text directly, then save"
              : "Click changes to reject, or switch to edit mode"}
          </span>
        </div>
      </div>
      <div
        ref={containerRef}
        style={containerStyle}
      >
        {isEditMode ? (
          <textarea
            ref={textareaRef}
            value={editableText}
            onChange={(event) => {
              setEditableText(event.target.value);
              setHasUnsavedChanges(true);
            }}
            style={editAreaStyle}
            spellCheck
          />
        ) : (
          renderContent()
        )}
      </div>
    </div>
  );
};

export default withStreamlitConnection(ClickableDiff);
