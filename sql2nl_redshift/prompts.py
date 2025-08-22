INSTRUCTION_TEMPLATE = (
    "You are a technical writer. Convert the following Amazon Redshift SQL query "
    "into a concise, plain-English explanation for an analyst. "
    "Mention any Redshift-specific features (QUALIFY, Spectrum external tables, DISTKEY/SORTKEY, system tables) if present. "
    "Be accurate and keep it under 2 sentences.\n\n"
    "SQL:\n{sql}\n\nExplanation:"
)
