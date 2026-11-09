import React from "react";
import { cn } from "@/lib/utils";

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  headerAction?: React.ReactNode;
}

export function Card({
  children,
  title,
  subtitle,
  headerAction,
  className,
  ...props
}: CardProps) {
  return (
    <div
      className={cn(
        "rounded-xl bg-surface border border-border overflow-hidden",
        className
      )}
      {...props}
    >
      {(title || headerAction) && (
        <div className="px-5 py-4 border-b border-border-subtle flex items-center justify-between">
          <div>
            {title && (
              <h3 className="text-sm font-medium tracking-tight text-neutral-200">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="text-xs text-neutral-500 mt-0.5">{subtitle}</p>
            )}
          </div>
          {headerAction && <div>{headerAction}</div>}
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  );
}

