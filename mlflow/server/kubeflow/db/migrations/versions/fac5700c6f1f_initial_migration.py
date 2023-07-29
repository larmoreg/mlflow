"""initial_migration

Revision ID: fac5700c6f1f
Revises:
Create Date: 2023-08-05 10:31:47.066106

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "fac5700c6f1f"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "experiment_permissions",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("experiment_id", sa.String(length=255), nullable=False),
        sa.Column("namespace", sa.String(length=255), nullable=False),
        sa.UniqueConstraint("experiment_id", "namespace", name="unique_experiment_namespace"),
    )
    op.create_table(
        "registered_model_permissions",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("namespace", sa.String(length=255), nullable=False),
        sa.UniqueConstraint("name", "namespace", name="unique_name_namespace"),
    )


def downgrade() -> None:
    op.drop_table("registered_model_permissions")
    op.drop_table("experiment_permissions")
