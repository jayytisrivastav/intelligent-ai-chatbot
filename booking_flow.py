import re
from datetime import date, datetime, time, timedelta
from booking_store import save_booking

def extract_details(message):
    """Read a day, start time, and duration from one message."""
    text = message.lower()
    details = {}

    # Understand "today", "tomorrow", or a date like 2026-11-05.
    if re.search(r"\btomorrow\b", text):
        details["day"] = date.today() + timedelta(days=1)
    elif re.search(r"\btoday\b", text):
        details["day"] = date.today()
    else:
        match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
        if match:
            try:
                details["day"] = date.fromisoformat(match.group())
            except ValueError:
                pass  # An invalid date stays missing, so the bot asks again.

    # Understand "2 PM", "2pm", or "2:30 PM".
    match = re.search(
        r"\b(1[0-2]|[1-9])(?::([0-5]\d))?\s*(am|pm)\b", text
    )
    if match:
        hour = int(match[1]) % 12
        if match[3] == "pm":
            hour += 12
        details["start"] = time(hour, int(match[2] or 0))

    # Understand "one hour", "2 hours", or "30 minutes".
    match = re.search(
        r"\b(one|two|three|an|a|\d{1,3})\s+"
        r"(hours?|hrs?|minutes?|mins?)\b", text
    )
    if match:
        numbers = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}
        number, unit = match.groups()
        amount = int(number) if number.isdigit() else numbers[number]
        minutes = amount * 60 if unit.startswith("h") else amount
        if 0 < minutes <= 24 * 60:
            details["minutes"] = minutes

    return details


class BookingFlow:
    def __init__(self):
        self.active = False
        self.awaiting_confirmation = False
        self.details = {}

    def begin(self, message):
        self.active = True
        self.awaiting_confirmation = False
        self.details = {}
        return self.reply(message)

    def reply(self, message):
        answer = message.strip().lower()

        if answer in {"cancel", "never mind", "nevermind"}:
            self.active = False
            return "Okay, I dropped this booking request."

        if self.awaiting_confirmation:
            if answer in {"yes", "y", "confirm"}:
                                start = datetime.combine(self.details["day"], self.details["start"])
            end = start + timedelta(minutes=self.details["minutes"])

            if start <= datetime.now():
                    self.active = False
                    return "That start time has passed. Please start a new booking."

            booking_id = save_booking(start, end)
            self.awaiting_confirmation = False

            if booking_id is None:
                    del self.details["start"]
                    return (
                        "That slot overlaps another booking. "
                        "What other start time would you like?"
                    )

            self.active = False
            return f"Microscope booked! Your booking ID is {booking_id}."
            return "Please type yes to confirm or no to discard this request."

        # Merge new details with the ones remembered from earlier turns.
        self.details.update(extract_details(message))

        if "day" not in self.details:
            return "Which day? You can say tomorrow or enter a date like 2026-11-05."
        if "start" not in self.details:
            return "What time would you like to start? For example, 2 PM."
        if "minutes" not in self.details:
            return "For how long? For example, one hour or 30 minutes."

        start = datetime.combine(self.details["day"], self.details["start"])
        end = start + timedelta(minutes=self.details["minutes"])
        self.awaiting_confirmation = True

        day = start.strftime("%A, %d %B %Y")
        start_time = start.strftime("%I:%M %p").lstrip("0")
        end_time = end.strftime("%I:%M %p").lstrip("0")
        if end.date() != start.date():
            end_time += end.strftime(" on %d %B")

        return (
            f"Confirm the microscope request for {day}, "
            f"{start_time} to {end_time}? (yes/no)"
        )