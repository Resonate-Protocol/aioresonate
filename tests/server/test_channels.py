"""Tests for channel routing infrastructure."""

from __future__ import annotations

from uuid import UUID

from aiosendspin.server.channels import MAIN_CHANNEL, ChannelRouter


class TestMainChannel:
    """Tests for MAIN_CHANNEL constant."""

    def test_main_channel_is_uuid(self) -> None:
        """MAIN_CHANNEL should be a UUID."""
        assert isinstance(MAIN_CHANNEL, UUID)

    def test_main_channel_is_consistent(self) -> None:
        """MAIN_CHANNEL should have a consistent value."""
        # Using the null UUID as the main channel identifier
        assert UUID("00000000-0000-0000-0000-000000000000") == MAIN_CHANNEL


class TestChannelRouterDefaults:
    """Tests for ChannelRouter default behavior."""

    def test_get_channel_returns_main_by_default(self) -> None:
        """Unassigned players should be on MAIN_CHANNEL."""
        router = ChannelRouter()
        assert router.get_channel("player-1") == MAIN_CHANNEL

    def test_get_players_on_main_channel_empty_initially(self) -> None:
        """No players on MAIN_CHANNEL until explicitly assigned."""
        router = ChannelRouter()
        # Players that haven't been assigned are implicitly on MAIN
        # but get_players_on_channel only returns explicitly assigned players
        assert router.get_players_on_channel(MAIN_CHANNEL) == []


class TestChannelRouterAssignment:
    """Tests for ChannelRouter assignment behavior."""

    def test_set_channel_assigns_player(self) -> None:
        """set_channel should assign player to specified channel."""
        router = ChannelRouter()
        custom_channel = UUID("11111111-1111-1111-1111-111111111111")

        router.set_channel("player-1", custom_channel)

        assert router.get_channel("player-1") == custom_channel

    def test_set_channel_to_main_explicitly(self) -> None:
        """Players can be explicitly assigned to MAIN_CHANNEL."""
        router = ChannelRouter()

        router.set_channel("player-1", MAIN_CHANNEL)

        assert router.get_channel("player-1") == MAIN_CHANNEL
        assert "player-1" in router.get_players_on_channel(MAIN_CHANNEL)

    def test_get_players_on_channel_returns_assigned_players(self) -> None:
        """get_players_on_channel should return all players on that channel."""
        router = ChannelRouter()
        channel_a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

        router.set_channel("player-1", channel_a)
        router.set_channel("player-2", channel_a)
        router.set_channel("player-3", MAIN_CHANNEL)

        players_on_a = router.get_players_on_channel(channel_a)
        assert len(players_on_a) == 2
        assert "player-1" in players_on_a
        assert "player-2" in players_on_a

    def test_reassign_player_to_different_channel(self) -> None:
        """Reassigning a player should update their channel."""
        router = ChannelRouter()
        channel_a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        channel_b = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

        router.set_channel("player-1", channel_a)
        assert router.get_channel("player-1") == channel_a

        router.set_channel("player-1", channel_b)
        assert router.get_channel("player-1") == channel_b
        assert "player-1" not in router.get_players_on_channel(channel_a)
        assert "player-1" in router.get_players_on_channel(channel_b)


class TestChannelRouterRemoval:
    """Tests for removing players from channels."""

    def test_remove_player_returns_to_implicit_main(self) -> None:
        """Removing a player should return them to implicit MAIN_CHANNEL."""
        router = ChannelRouter()
        channel_a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

        router.set_channel("player-1", channel_a)
        router.remove_player("player-1")

        # Back to implicit MAIN (not explicitly assigned)
        assert router.get_channel("player-1") == MAIN_CHANNEL
        assert "player-1" not in router.get_players_on_channel(channel_a)

    def test_remove_nonexistent_player_is_noop(self) -> None:
        """Removing a player that isn't assigned should not raise."""
        router = ChannelRouter()
        router.remove_player("nonexistent")  # Should not raise
