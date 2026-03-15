import { SignUp } from "@clerk/nextjs";

export default function SignUpPage() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <SignUp
        appearance={{
          elements: {
            rootBox: "mx-auto",
            card: "bg-card border border-border shadow-lg",
            headerTitle: "text-foreground",
            headerSubtitle: "text-muted-foreground",
            formButtonPrimary:
              "bg-primary text-primary-foreground hover:bg-primary/90",
            footerActionLink: "text-primary hover:text-primary/80",
            formFieldInput:
              "bg-background border-border text-foreground",
            formFieldLabel: "text-foreground",
            dividerLine: "bg-border",
            dividerText: "text-muted-foreground",
            socialButtonsBlockButton:
              "border-border text-foreground hover:bg-accent",
          },
        }}
        fallbackRedirectUrl="/explore"
        signInUrl="/sign-in"
      />
    </div>
  );
}
