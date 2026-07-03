# Auto-start a TORCS practice race via GUI automation

import time
import pyautogui

try:
    import pygetwindow as gw
except ImportError:
    gw = None

_TITLE_HINTS = ("torcs", "wtorcs", "speed dreams", "scr")
POST_FOCUS_SETTLE = 2.5


def _win32_force_foreground(title_substr: str) -> bool:
    try:
        import win32gui
        import win32con
    except ImportError:
        return False

    matches = []

    def _enum(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            t = win32gui.GetWindowText(hwnd)
            if t and title_substr in t.lower():
                matches.append(hwnd)

    win32gui.EnumWindows(_enum, None)
    if not matches:
        return False
    hwnd = matches[0]
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.3)
        return True
    except Exception:
        return False


def focus_torcs_window(timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    seen_titles = set()

    while time.time() < deadline:
        if gw is not None:
            try:
                all_titles = [t for t in gw.getAllTitles() if t]
            except Exception:
                all_titles = []
            seen_titles.update(all_titles)
            titles = [t for t in all_titles
                      if any(h in t.lower() for h in _TITLE_HINTS)]
            if titles:
                try:
                    win = gw.getWindowsWithTitle(titles[0])[0]
                    try:
                        win.minimize(); time.sleep(0.5)
                        win.restore();  time.sleep(0.8)
                    except Exception:
                        pass
                    try:
                        win.activate()
                    except Exception:
                        pass
                    time.sleep(0.3)
                    return True
                except Exception:
                    pass

        for hint in _TITLE_HINTS:
            if _win32_force_foreground(hint):
                return True

        time.sleep(0.5)

    return False


def main():
    time.sleep(10.0)
    focus_torcs_window()
    time.sleep(POST_FOCUS_SETTLE)

    focus_torcs_window(timeout=5.0)
    time.sleep(1.0)

    actions = ['enter', 'down', 'down', 'down', 'down', 'down', 'enter', 'enter']
    for key in actions:
        pyautogui.press(key)
        time.sleep(0.3)


if __name__ == '__main__':
    main()
