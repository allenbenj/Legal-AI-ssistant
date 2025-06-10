# Design System Overview

This project uses a small set of design tokens and React components to keep styling consistent.

## Tokens

Token values are defined in `frontend/src/design-system/tokens.ts` and include colors, spacing, typography, border radius, shadows and animation speeds. Import tokens into components to avoid magic values.

Example:
```ts
import { colors, spacing } from '../design-system/tokens';
```

## Components

Core UI components live in `frontend/src/design-system/components`.
Currently available:
- `Button` &ndash; supports `primary`, `secondary` and `outline` variants with `sm`, `md`, `lg` sizes.
- `Input` &ndash; basic text input using tokenised spacing and borders.
- `Card` &ndash; simple container with padding and shadow.
- `Alert` &ndash; color coded status messages.
- `Grid` &ndash; layout utility for building responsive grids.

These components automatically apply token values so usage is consistent across the application.

## Usage Example

```tsx
import { Button } from '../design-system';

<Button variant="primary" size="md">Save</Button>
```

When referencing the design system from code outside of the `frontend/src`
directory (for example, inside the `legal_ai_system` package), use a relative
import path:

```tsx
import { Button } from '../../frontend/src/design-system';
```

## Extending

When building new features, prefer using these base components or create additional ones in the same folder referencing token values. This keeps styles unified and reduces duplication.
